# Job execution backends and per-process run contexts.
#
# The coordinator submits screening evaluations and whole local searches as
# jobs. Each job runs `TikTak` code on the executing process, which looks up the
# run context (objectives, transform, configuration, store handle) installed
# there before the search started. Completed jobs are reported on a channel so
# the coordinator can react to whichever finishes first.

"""
    AbstractExecutor

Backend that runs jobs for the coordinator. `capacity(ex)` is the number of
jobs that may be in flight; `submit!(ex, done, tag, f, args...)` starts
`f(args...)` and eventually puts `(tag, (:ok, result))` or `(tag, (:error, exception))`
on the channel `done`.
"""
abstract type AbstractExecutor end

"""Run jobs one at a time on the coordinator; fully reproducible with a fixed seed."""
struct InlineExecutor <: AbstractExecutor end

"""
    ThreadedExecutor(capacity=Threads.nthreads())

Run up to `capacity` jobs concurrently as tasks on the coordinator's threads.
The objective must be thread-safe.
"""
struct ThreadedExecutor <: AbstractExecutor
    capacity::Int
    function ThreadedExecutor(capacity::Integer=Threads.nthreads())
        capacity >= 1 || throw(ArgumentError("capacity must be positive"))
        return new(Int(capacity))
    end
end

"""
    DistributedExecutor(pids=Distributed.workers())

Run one job at a time on each of the worker processes `pids`. Every worker must
have `TikTak`, the objective, and any local-solver package loaded (typically
via `@everywhere`).
"""
struct DistributedExecutor <: AbstractExecutor
    pids::Vector{Int}
    free::Channel{Int}
    function DistributedExecutor(pids::AbstractVector{<:Integer}=Distributed.workers())
        pids = unique(Int.(pids))
        isempty(pids) && throw(ArgumentError("DistributedExecutor needs at least one worker process"))
        myid() in pids && throw(ArgumentError("workers must be worker processes, not the coordinator ($(myid()))"))
        missing = setdiff(pids, procs())
        isempty(missing) || throw(ArgumentError("unknown worker processes: $missing"))
        free = Channel{Int}(length(pids))
        foreach(p -> put!(free, p), pids)
        return new(pids, free)
    end
end

capacity(::InlineExecutor) = 1
capacity(ex::ThreadedExecutor) = ex.capacity
capacity(ex::DistributedExecutor) = length(ex.pids)

_outcome(f, args) = try
    (:ok, f(args...))
catch exc
    (:error, exc)
end

function submit!(::InlineExecutor, done::Channel, tag, f, args...)
    put!(done, (tag, _outcome(f, args)))
    return nothing
end

function submit!(::ThreadedExecutor, done::Channel, tag, f, args...)
    Threads.@spawn put!(done, (tag, _outcome(f, args)))
    return nothing
end

function submit!(ex::DistributedExecutor, done::Channel, tag, f, args...)
    pid = take!(ex.free)
    @async begin
        outcome = _outcome(remotecall_fetch, (f, pid, args...))
        put!(ex.free, pid)
        put!(done, (tag, outcome))
    end
    return nothing
end

# --- Run contexts -------------------------------------------------------------

struct Context{O,P,S}
    objective::O
    screen_objective::P         # `nothing` unless the run is staged
    transform::BoxTransform
    config
    store::S
end

const CONTEXTS = Dict{String,Context}()
const CONTEXTS_LOCK = ReentrantLock()

function _install_context!(run_id::AbstractString, objective, screen_objective, transform, config, store)
    lock(CONTEXTS_LOCK) do
        CONTEXTS[String(run_id)] = Context(objective, screen_objective, transform, config, store)
    end
    return nothing
end

function _remove_context!(run_id::AbstractString)
    lock(CONTEXTS_LOCK) do
        delete!(CONTEXTS, String(run_id))
    end
    return nothing
end

function _context(run_id::AbstractString)
    lock(CONTEXTS_LOCK) do
        ctx = get(CONTEXTS, String(run_id), nothing)
        ctx === nothing && throw(ErrorException("no active TikTak run $run_id on process $(myid())"))
        return ctx
    end
end

function setup!(::Union{InlineExecutor,ThreadedExecutor}, run_id, objective, screen_objective, transform, config,
                store)
    _install_context!(run_id, objective, screen_objective, transform, config, store)
    return nothing
end

function setup!(ex::DistributedExecutor, run_id, objective, screen_objective, transform, config, store)
    handle = RemoteStore(myid(), String(run_id))
    for pid in ex.pids
        try
            remotecall_fetch(_install_context!, pid, run_id, objective, screen_objective, transform, config,
                             handle)
        catch exc
            throw(ErrorException("could not install the run on worker $pid; make sure every worker " *
                                 "has run `using TikTak` and defines the objective (`@everywhere`). " *
                                 "Underlying error: " * sprint(showerror, exc)))
        end
    end
    return nothing
end

function teardown!(::Union{InlineExecutor,ThreadedExecutor}, run_id)
    _remove_context!(run_id)
    return nothing
end

function teardown!(ex::DistributedExecutor, run_id)
    for pid in ex.pids
        try
            remotecall_fetch(_remove_context!, pid, run_id)
        catch
            # A dead worker cannot be cleaned up; the coordinator's own state is unaffected.
        end
    end
    return nothing
end

# --- Jobs ---------------------------------------------------------------------

function _screen_job(run_id, index, unit)
    ctx = _context(run_id)
    staged = ctx.screen_objective !== nothing
    return evaluate_point(staged ? ctx.screen_objective : ctx.objective, ctx.transform, ctx.store, unit,
                          "screen:$index", ctx.config; screening=staged)
end

function _local_job(run_id, index)
    ctx = _context(run_id)
    return run_local(ctx.objective, ctx.transform, ctx.store, index, ctx.config)
end
