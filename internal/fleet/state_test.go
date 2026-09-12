package fleet

import (
	"errors"
	"testing"
)

func TestHappyPathProvisioning(t *testing.T) {
	want := []struct {
		event Event
		phase Phase
	}{
		{EventProvision, PhaseProvisioning},
		{EventImageWritten, PhaseDriverInstall},
		{EventDriversReady, PhaseValidating},
		{EventValidationPass, PhaseReady},
	}

	cur := PhaseDiscovered
	for i, step := range want {
		next, err := Transition(cur, step.event)
		if err != nil {
			t.Fatalf("step %d (%s from %s): %v", i, step.event, cur, err)
		}
		if next != step.phase {
			t.Fatalf("step %d: %s + %s = %s, want %s", i, cur, step.event, next, step.phase)
		}
		cur = next
	}
	if !Schedulable(cur) {
		t.Errorf("host ended in %s, which is not schedulable", cur)
	}
}

func TestIllegalTransitionsAreRejected(t *testing.T) {
	cases := []struct {
		name  string
		from  Phase
		event Event
	}{
		{"cannot validate a freshly discovered host", PhaseDiscovered, EventValidationPass},
		{"cannot skip driver install", PhaseProvisioning, EventDriversReady},
		{"cannot drain a host that was never ready", PhaseDiscovered, EventDrained},
		{"cannot repair straight to ready", PhaseRepair, EventValidationPass},
		{"cannot provision a ready host", PhaseReady, EventProvision},
		{"cannot retry a ready host", PhaseReady, EventRetry},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, err := Transition(c.from, c.event)
			if err == nil {
				t.Fatalf("%s + %s was allowed", c.from, c.event)
			}
			var ill ErrIllegalTransition
			if !errors.As(err, &ill) {
				t.Fatalf("error type = %T, want ErrIllegalTransition", err)
			}
		})
	}
}

// The invariant that matters most: a host carrying live inference traffic is
// never pulled straight out for repair. It drains first.
func TestUnhealthyReadyHostDrainsBeforeRepair(t *testing.T) {
	next, err := Transition(PhaseReady, EventHealthLost)
	if err != nil {
		t.Fatal(err)
	}
	if next != PhaseDraining {
		t.Fatalf("Ready + HealthLost = %s, want Draining — live traffic must drain first", next)
	}
	if Schedulable(next) {
		t.Error("Draining reported schedulable; new workloads would land on a dying host")
	}
	if Can(PhaseReady, EventDrained) {
		t.Error("Ready accepts Drained directly, bypassing the drain state")
	}
}

// A repaired host must re-validate. Trusting a repair report and going straight
// back to Ready is how a bad GPU re-enters the pool.
func TestRepairedHostMustRevalidate(t *testing.T) {
	next, err := Transition(PhaseRepair, EventRepaired)
	if err != nil {
		t.Fatal(err)
	}
	if next != PhaseValidating {
		t.Fatalf("Repair + Repaired = %s, want Validating", next)
	}
}

// Queues redeliver and controllers re-reconcile, so the same event arriving
// twice must not break the machine.
func TestRedeliveredEventIsSafe(t *testing.T) {
	// Draining + HealthLost is explicitly a self-loop: the host is already on
	// its way out, a second health alarm changes nothing.
	next, err := Transition(PhaseDraining, EventHealthLost)
	if err != nil {
		t.Fatalf("redelivered HealthLost in Draining: %v", err)
	}
	if next != PhaseDraining {
		t.Fatalf("Draining + HealthLost = %s, want Draining", next)
	}

	again, err := Transition(next, EventHealthLost)
	if err != nil || again != PhaseDraining {
		t.Fatalf("third delivery: phase=%s err=%v", again, err)
	}
}

func TestTerminalStateAcceptsNothing(t *testing.T) {
	if !IsTerminal(PhaseDecommissioned) {
		t.Fatal("Decommissioned should be terminal")
	}
	for _, e := range []Event{EventProvision, EventRetry, EventRepaired, EventDecommission} {
		if _, err := Transition(PhaseDecommissioned, e); err == nil {
			t.Errorf("Decommissioned accepted %s", e)
		}
	}
}

// Failed is deliberately *not* terminal — a failed host can be retried. But it
// must take an explicit Retry to do it, never an implicit re-provision.
func TestFailedHostIsRetryableButNotAutomatic(t *testing.T) {
	if IsTerminal(PhaseFailed) {
		t.Fatal("Failed should not be terminal; hosts get retried")
	}
	next, err := Transition(PhaseFailed, EventRetry)
	if err != nil {
		t.Fatal(err)
	}
	if next != PhaseProvisioning {
		t.Fatalf("Failed + Retry = %s, want Provisioning", next)
	}
	if Can(PhaseFailed, EventImageWritten) {
		t.Error("Failed accepts ImageWritten, so a stale in-flight event could resurrect it silently")
	}
}

func TestOnlyReadyIsSchedulable(t *testing.T) {
	for _, p := range Phases() {
		want := p == PhaseReady
		if got := Schedulable(p); got != want {
			t.Errorf("Schedulable(%s) = %v, want %v", p, got, want)
		}
	}
}

// Every phase must be reachable from Discovered, or the table has dead states.
func TestEveryPhaseIsReachableFromDiscovered(t *testing.T) {
	for _, p := range Phases() {
		if !Reachable(PhaseDiscovered, p) {
			t.Errorf("%s is unreachable from Discovered — dead state in the table", p)
		}
	}
}

// Every non-terminal phase must offer a way out, or a host can wedge forever.
func TestNoPhaseIsADeadEnd(t *testing.T) {
	for _, p := range Phases() {
		if IsTerminal(p) {
			continue
		}
		if len(EventsFrom(p)) == 0 {
			t.Errorf("%s is non-terminal with no legal events — hosts wedge here", p)
		}
	}
}

// Every non-terminal phase must be able to reach Decommissioned, or a host can
// never be removed from the fleet.
func TestEveryPhaseCanReachDecommissioned(t *testing.T) {
	for _, p := range Phases() {
		if IsTerminal(p) {
			continue
		}
		if !Reachable(p, PhaseDecommissioned) {
			t.Errorf("a host in %s can never be decommissioned", p)
		}
	}
}

func TestEventsFromIsSortedAndComplete(t *testing.T) {
	got := EventsFrom(PhaseReady)
	if len(got) != 2 {
		t.Fatalf("EventsFrom(Ready) = %v, want 2 events", got)
	}
	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Fatalf("EventsFrom not sorted: %v", got)
		}
	}
}

func TestTableVersionIsSet(t *testing.T) {
	if TableVersion < 1 {
		t.Fatal("TableVersion must be a positive integer")
	}
}
