// Package fleet models the lifecycle of a physical GPU host as an explicit,
// versioned state machine.
//
// The reason this is a table and not a pile of if-statements: provisioning
// spans minutes to hours, crosses a BMC, a PXE boot, a driver install and a
// health check, and gets interrupted. A controller that infers "what should
// happen next" from ad-hoc field checks will, eventually, decommission a node
// that was merely rebooting. Here every legal move is enumerated, everything
// else is an error, and the table carries a version so a stored state written
// by an older build is detectable rather than silently misread.
package fleet

import (
	"fmt"
	"sort"
)

// TableVersion is bumped whenever the transition table changes shape. Persisted
// host records carry the version they were written under.
const TableVersion = 2

// Phase is a host lifecycle state.
type Phase string

const (
	// PhaseDiscovered - the BMC answered, we know the host exists, nothing else.
	PhaseDiscovered Phase = "Discovered"
	// PhaseProvisioning - OS image being written (PXE/iPXE).
	PhaseProvisioning Phase = "Provisioning"
	// PhaseDriverInstall - OS up, installing GPU driver and CUDA stack.
	PhaseDriverInstall Phase = "DriverInstall"
	// PhaseValidating - burn-in: GPU count, ECC state, NCCL all-reduce, link width.
	PhaseValidating Phase = "Validating"
	// PhaseReady - in the pool, schedulable.
	PhaseReady Phase = "Ready"
	// PhaseDraining - being emptied of workloads, not schedulable.
	PhaseDraining Phase = "Draining"
	// PhaseRepair - drained and handed to remediation.
	PhaseRepair Phase = "Repair"
	// PhaseDecommissioned - out of the fleet. Terminal.
	PhaseDecommissioned Phase = "Decommissioned"
	// PhaseFailed - provisioning or validation gave up. Terminal until a human
	// or a policy retries it explicitly.
	PhaseFailed Phase = "Failed"
)

// Event is something that happened to a host, or something we decided to do.
type Event string

const (
	EventProvision      Event = "Provision"
	EventImageWritten   Event = "ImageWritten"
	EventDriversReady   Event = "DriversReady"
	EventValidationPass Event = "ValidationPass"
	EventValidationFail Event = "ValidationFail"
	EventHealthLost     Event = "HealthLost"
	EventDrained        Event = "Drained"
	EventRepaired       Event = "Repaired"
	EventDecommission   Event = "Decommission"
	EventRetry          Event = "Retry"
)

type edge struct {
	from  Phase
	event Event
}

// transitions is the whole machine. If a (phase, event) pair is not in here it
// cannot happen, and Transition says so rather than guessing.
var transitions = map[edge]Phase{
	{PhaseDiscovered, EventProvision}:         PhaseProvisioning,
	{PhaseDiscovered, EventDecommission}:      PhaseDecommissioned,
	{PhaseProvisioning, EventImageWritten}:    PhaseDriverInstall,
	{PhaseProvisioning, EventValidationFail}:  PhaseFailed,
	{PhaseDriverInstall, EventDriversReady}:   PhaseValidating,
	{PhaseDriverInstall, EventValidationFail}: PhaseFailed,
	{PhaseValidating, EventValidationPass}:    PhaseReady,
	{PhaseValidating, EventValidationFail}:    PhaseFailed,

	// A ready host that loses health drains first. It is never yanked straight
	// to Repair — there is live inference traffic on it.
	{PhaseReady, EventHealthLost}:    PhaseDraining,
	{PhaseReady, EventDecommission}:  PhaseDraining,
	{PhaseDraining, EventDrained}:    PhaseRepair,
	{PhaseDraining, EventHealthLost}: PhaseDraining, // already going; stay put

	// Repair either returns the host to validation (never straight to Ready —
	// it has to earn its way back in) or ends its life.
	{PhaseRepair, EventRepaired}:       PhaseValidating,
	{PhaseRepair, EventDecommission}:   PhaseDecommissioned,
	{PhaseRepair, EventValidationFail}: PhaseFailed,

	{PhaseFailed, EventRetry}:        PhaseProvisioning,
	{PhaseFailed, EventDecommission}: PhaseDecommissioned,
}

// terminal states accept nothing further.
var terminal = map[Phase]bool{
	PhaseDecommissioned: true,
}

// IsTerminal reports whether a phase accepts no further events.
func IsTerminal(p Phase) bool { return terminal[p] }

// Schedulable reports whether workloads may be placed on a host in this phase.
// Only Ready is schedulable — in particular Draining is not, which is the whole
// reason draining is its own state.
func Schedulable(p Phase) bool { return p == PhaseReady }

// ErrIllegalTransition is returned for any (phase, event) pair not in the table.
type ErrIllegalTransition struct {
	From  Phase
	Event Event
}

func (e ErrIllegalTransition) Error() string {
	return fmt.Sprintf("illegal transition: cannot %s a host in %s (table v%d)", e.Event, e.From, TableVersion)
}

// Transition applies an event to a phase.
//
// Re-delivery is normal: a queue redelivers, a controller re-reconciles, an
// activity retries. So an event that would move a host to the phase it is
// already in succeeds as a no-op rather than erroring. Everything genuinely
// illegal returns ErrIllegalTransition.
func Transition(from Phase, e Event) (Phase, error) {
	if IsTerminal(from) {
		return from, ErrIllegalTransition{From: from, Event: e}
	}
	to, ok := transitions[edge{from, e}]
	if !ok {
		return from, ErrIllegalTransition{From: from, Event: e}
	}
	return to, nil
}

// Can reports whether an event is legal in a phase, without applying it.
func Can(from Phase, e Event) bool {
	_, err := Transition(from, e)
	return err == nil
}

// EventsFrom lists the legal events in a phase, sorted. Useful for surfacing
// "what can I do with this host" through the API instead of making callers
// guess.
func EventsFrom(p Phase) []Event {
	var out []Event
	for edge := range transitions {
		if edge.from == p {
			out = append(out, edge.event)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Phases lists every phase, sorted.
func Phases() []Phase {
	seen := map[Phase]bool{}
	for e, to := range transitions {
		seen[e.from] = true
		seen[to] = true
	}
	out := make([]Phase, 0, len(seen))
	for p := range seen {
		out = append(out, p)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Reachable reports whether any sequence of events leads from `from` to `to`.
// A phase that cannot be reached from Discovered is dead code in the table, and
// the test suite asserts there is none.
func Reachable(from, to Phase) bool {
	if from == to {
		return true
	}
	seen := map[Phase]bool{from: true}
	queue := []Phase{from}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for e, next := range transitions {
			if e.from != cur || seen[next] {
				continue
			}
			if next == to {
				return true
			}
			seen[next] = true
			queue = append(queue, next)
		}
	}
	return false
}
