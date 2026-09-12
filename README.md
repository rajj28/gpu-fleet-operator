# gpu-fleet-operator

A control plane for a GPU inference fleet: a **versioned host lifecycle state machine**, a
**durable Temporal provisioning workflow**, and a **bin-packer that measures and repairs
fragmentation** — the capacity you own but cannot schedule.

**35 tests, `go vet` clean.** Coverage: `fleet` 95.3%, `scheduler` 91.4%.
The Temporal workflows are tested against the SDK's replay environment, so no server is needed
to run the suite.

```
go test ./... -cover
ok  .../internal/fleet       coverage: 95.3% of statements
ok  .../internal/scheduler   coverage: 91.4% of statements
ok  .../internal/workflow    coverage: 49.1% of statements
```

---

## The problem this is about

A fleet can report **30% free GPUs and still refuse an 8-GPU job**, because that 30% is scattered
one or two GPUs at a time across forty hosts. Utilization does not surface it. You find out when a
customer's deployment pends forever on a cluster you know has room.

So `scheduler` exposes the number utilization hides:

```go
// Four 8-GPU nodes, six GPUs used on each.
f.Utilization()              // 0.75   — looks healthy
f.TotalFree()                // 8      — a whole node's worth, on paper
f.Fragmentation(h100, 8)     // 1.00   — none of it can take an 8-GPU job
f.LargestPlaceable(h100)     // 2
```

`Fragmentation(type, size)` is the share of free GPUs that cannot accept a workload that wide,
because they sit on nodes with too little room. Then `Defragment` turns that into a plan:

```
defrag h100 size=8: 6 moves, 12 GPU relocated, 3 nodes freed,
                    fragmentation 1.00 -> 0.00, largest 6 -> 8
```

Plans are computed against a **clone**, so planning never mutates the live fleet, and carry
before/after numbers so a human or a policy can decide whether the disruption is worth it.
`Plan.Worthwhile()` reports whether it buys anything at all.

---

## Three design decisions, and why

**Evacuate the emptiest nodes, and never partially.** A node holding one 2-GPU replica is cheap to
clear and yields a whole node; one holding six is expensive and yields the same node. So candidates
are sorted by how little they hold. And a node is only evacuated if *every* workload on it can be
rehomed — a half-evacuation pays the disruption and still leaves the node occupied, which is
strictly worse than doing nothing. `TestDefragmentDoesNotPartiallyEvacuate` pins that.

**`Apply` is all-or-nothing, and detects drift.** A plan records where each workload was when the
plan was made. If anything moved in between, `Apply` refuses rather than acting on a stale premise —
the exact race a reconcile loop hits in production. Any mid-plan failure rolls the fleet back to
where it started.

**The state machine is a table, not a pile of if-statements.** Provisioning spans minutes to hours
across a BMC, a PXE boot, a driver install and an NCCL burn-in, and it gets interrupted. A
controller that infers "what should happen next" from ad-hoc field checks will eventually
decommission a node that was merely rebooting. Here every legal move is enumerated, everything else
returns `ErrIllegalTransition`, and the table carries a `TableVersion` so state written by an older
build is detectable rather than silently misread.

The invariants the tests actually enforce:

- A `Ready` host that loses health **drains before repair** — there is live traffic on it.
- A repaired host **re-validates**; it never returns straight to `Ready` on a repair report.
- `Failed` is retryable, but only via an explicit `Retry` — a stale in-flight event cannot resurrect it.
- Every phase is reachable from `Discovered`, no phase is a dead end, and every non-terminal phase
  can reach `Decommissioned`. Those three are property tests over the table, so adding a phase
  without wiring it up fails CI.

---

## Durability

`ProvisionHostWorkflow` makes every step an activity, so every step is a checkpoint: if the worker
dies after `InstallDrivers` and before `ValidateGPUs`, the replacement worker resumes at
`ValidateGPUs` rather than re-imaging the box. Heartbeat timeouts catch an activity that is wedged
but still nominally running. `QueryPhase` answers "where is my host?" live, with no database read —
which is what makes a self-service API possible instead of a ticket.

One distinction that matters: **an unhealthy host is a result, not an error.** `ValidateGPUs`
returns a `ValidationReport` when burn-in finds a bad GPU, and an `error` only when burn-in could
not run. Folding those together is how "this host has 7 of 8 GPUs" gets retried four times by a
retry policy that cannot tell the difference between a broken host and a broken check.

---

## Two bugs the tests caught

Worth recording, because they are the reason the suite exists.

**The defragmenter churned a fully-packed node.** With `n1` full and `n2` empty, an 8-GPU job
already fits — nothing to do. The planner instead evacuated `n1` onto `n2`: eight GPUs of live
traffic relocated for zero gain. Two fixes: return early when `LargestPlaceable >= targetSize`, and
skip fully-allocated nodes as candidates, since fragmentation lives on *partially* used nodes and a
full node wastes nothing.

**The retry path issued an illegal transition.** After a failed attempt the host sits in `Failed`,
and the loop head unconditionally issued `Provision` — which is legal from `Discovered` and from
nowhere else. Every retry after the first died on `ErrIllegalTransition`. The state machine caught
what a pile of if-statements would have silently allowed. Now the loop picks `Provision` or `Retry`
based on the phase it is actually in, in one place.

---

## Layout

```
internal/scheduler/   bin-packing, fragmentation metric, defragmentation planner   (no deps)
internal/fleet/       host lifecycle state machine, versioned transition table     (no deps)
internal/workflow/    Temporal provisioning + drain workflows, activity interface
```

`scheduler` and `fleet` depend on nothing outside the standard library — the packing and the rules
are testable without a cluster, a Temporal server, or a mock of either. `workflow` owns sequence
and durability and defers every rule to `fleet`, so there is exactly one definition of what may
follow what.

`Provisioner` is an interface, so a second BMC vendor or a different cloud drops in without the
workflow learning about it — the decoupling a platform exists to provide.

## Not done yet

- **No Kubernetes controller.** The CRD and reconciler are the obvious next layer: the state machine
  and the packer are the hard parts and they are done, but `InferenceCluster` as a CRD with a
  controller-runtime reconcile loop is not written.
- `workflow` coverage is 49% — the happy path, retry exhaustion, transient recovery, abort and drain
  are covered; the decommission branches are not.
- Topology is modelled as a `Domain` string on each node but the packer does not yet prefer
  same-domain placement for multi-GPU workloads. NVLink islands deserve real treatment.
- No real `Provisioner` implementation — the interface is there, PXE/Redfish/IPMI behind it is not.

## Licence

MIT.

## CI

`ci.yml.example` is a ready GitHub Actions workflow (`go vet` + `go test -race -cover`).
Move it to `.github/workflows/ci.yml` from the GitHub web UI — the CLI token used to push this
repo deliberately lacks `workflow` scope.
