// Package workflow drives host provisioning as a durable Temporal workflow.
//
// Provisioning a GPU host takes tens of minutes and touches a BMC, a PXE boot,
// a driver install and an NCCL burn-in. Any of those can hang, and the process
// running the orchestration will itself be redeployed halfway through. A cron
// job or a long-lived goroutine loses its place when that happens; a Temporal
// workflow resumes from its last completed activity, because its progress is
// event-sourced rather than held in memory.
//
// The workflow owns *sequence and durability*. It does not own the rules — legal
// transitions come from internal/fleet, so the state machine is unit-testable
// without a Temporal server and there is exactly one definition of what may
// follow what.
package workflow

import (
	"fmt"
	"time"

	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"

	"github.com/rajj28/gpu-fleet-operator/internal/fleet"
)

// TaskQueue is the queue provisioning workers listen on.
const TaskQueue = "gpu-fleet-provisioning"

// Signals and queries the control plane uses to talk to a running workflow.
const (
	// SignalAbort asks the workflow to stop and hand the host to decommission.
	SignalAbort = "abort"
	// QueryPhase returns the host's current lifecycle phase. This is how the
	// self-service API answers "where is my host?" without a database read.
	QueryPhase = "phase"
)

// ProvisionRequest is the manifest for one host bring-up.
type ProvisionRequest struct {
	HostID  string
	GPUType string
	GPUs    int
	Image   string
	// BurnInDuration is how long the NCCL/ECC validation runs.
	BurnInDuration time.Duration
}

// ProvisionResult is what the workflow returns once the host settles.
type ProvisionResult struct {
	HostID      string
	FinalPhase  fleet.Phase
	Attempts    int
	Transitions []string
}

// MaxProvisionAttempts caps how many times a failed host is retried before it
// is left in Failed for a human. Retrying forever on a host with a dead GPU
// burns hours of fleet capacity and hides the fault.
const MaxProvisionAttempts = 3

// ProvisionHostWorkflow takes a discovered host to Ready, or to Failed.
//
// Every step is an activity, so every step is a durable checkpoint: if the
// worker dies after InstallDrivers and before ValidateGPUs, the replacement
// worker resumes at ValidateGPUs rather than re-imaging the box.
func ProvisionHostWorkflow(ctx workflow.Context, req ProvisionRequest) (*ProvisionResult, error) {
	log := workflow.GetLogger(ctx)
	res := &ProvisionResult{HostID: req.HostID}
	phase := fleet.PhaseDiscovered

	// Expose the phase to queries, and keep it current as we advance.
	if err := workflow.SetQueryHandler(ctx, QueryPhase, func() (string, error) {
		return string(phase), nil
	}); err != nil {
		return nil, fmt.Errorf("register query handler: %w", err)
	}

	advance := func(e fleet.Event) error {
		next, err := fleet.Transition(phase, e)
		if err != nil {
			// An illegal transition is a bug in the workflow, not a flaky host.
			// Fail non-retryably so it surfaces instead of spinning.
			return temporal.NewNonRetryableApplicationError(
				err.Error(), "IllegalTransition", err)
		}
		res.Transitions = append(res.Transitions, fmt.Sprintf("%s--%s-->%s", phase, e, next))
		log.Info("host transition", "host", req.HostID, "from", phase, "event", e, "to", next)
		phase = next
		return nil
	}

	// An abort signal takes effect at the next step boundary rather than
	// mid-activity, so we never leave a half-written image behind.
	aborted := false
	workflow.Go(ctx, func(gctx workflow.Context) {
		workflow.GetSignalChannel(gctx, SignalAbort).Receive(gctx, nil)
		aborted = true
	})

	// Infrastructure activities are slow and side-effecting. Heartbeat timeouts
	// catch a wedged activity that is still technically "running", and the
	// backoff is deliberately patient — hammering a BMC does not help.
	slow := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 30 * time.Minute,
		HeartbeatTimeout:    2 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    15 * time.Second,
			BackoffCoefficient: 2.0,
			MaximumInterval:    5 * time.Minute,
			MaximumAttempts:    4,
			// A host that is physically absent will not appear on retry.
			NonRetryableErrorTypes: []string{"HostNotFound", "IllegalTransition"},
		},
	})
	quick := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 2 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    5 * time.Second,
			BackoffCoefficient: 2.0,
			MaximumAttempts:    5,
		},
	})

	for attempt := 1; attempt <= MaxProvisionAttempts; attempt++ {
		res.Attempts = attempt

		if aborted {
			return res, decommission(ctx, quick, req, res, advance, &phase)
		}

		// Entering an attempt, the host is either freshly Discovered (first time
		// round) or sitting in Failed from the previous attempt. Those need
		// different events: Provision moves a discovered host forward, Retry is
		// the only legal way out of Failed. Issuing Provision unconditionally
		// here is an illegal transition on every attempt after the first.
		entry := fleet.EventProvision
		if phase == fleet.PhaseFailed {
			entry = fleet.EventRetry
		}
		if err := advance(entry); err != nil {
			return res, err
		}
		if err := workflow.ExecuteActivity(slow, WriteImage, req).Get(ctx, nil); err != nil {
			if failAttempt(advance, res, &phase, err) != nil {
				return res, err
			}
			continue
		}
		if err := advance(fleet.EventImageWritten); err != nil {
			return res, err
		}

		if err := workflow.ExecuteActivity(slow, InstallDrivers, req).Get(ctx, nil); err != nil {
			if failAttempt(advance, res, &phase, err) != nil {
				return res, err
			}
			continue
		}
		if err := advance(fleet.EventDriversReady); err != nil {
			return res, err
		}

		var report ValidationReport
		if err := workflow.ExecuteActivity(slow, ValidateGPUs, req).Get(ctx, &report); err != nil {
			if failAttempt(advance, res, &phase, err) != nil {
				return res, err
			}
			continue
		}
		if !report.Healthy {
			log.Warn("validation failed", "host", req.HostID, "reason", report.Reason)
			if err := advance(fleet.EventValidationFail); err != nil {
				return res, err
			}
			if attempt < MaxProvisionAttempts {
				continue // loop head issues Retry
			}
			res.FinalPhase = phase
			return res, nil
		}

		if err := advance(fleet.EventValidationPass); err != nil {
			return res, err
		}
		if err := workflow.ExecuteActivity(quick, JoinPool, req).Get(ctx, nil); err != nil {
			return res, fmt.Errorf("host validated but could not join the pool: %w", err)
		}

		res.FinalPhase = phase
		return res, nil
	}

	res.FinalPhase = phase
	return res, nil
}

// failAttempt records a failed step, leaving the host in Failed. The loop head
// issues the Retry that moves it back to Provisioning, so there is exactly one
// place that decides how an attempt begins.
func failAttempt(advance func(fleet.Event) error, res *ProvisionResult, phase *fleet.Phase, cause error) error {
	if err := advance(fleet.EventValidationFail); err != nil {
		return err
	}
	if res.Attempts >= MaxProvisionAttempts {
		res.FinalPhase = *phase
	}
	return nil
}

func decommission(ctx workflow.Context, opts workflow.Context, req ProvisionRequest,
	res *ProvisionResult, advance func(fleet.Event) error, phase *fleet.Phase) error {
	if !fleet.Can(*phase, fleet.EventDecommission) {
		// Mid-provision there is no direct route out; fail first, then leave.
		if err := advance(fleet.EventValidationFail); err != nil {
			return err
		}
	}
	if err := advance(fleet.EventDecommission); err != nil {
		return err
	}
	if err := workflow.ExecuteActivity(opts, ReleaseHost, req).Get(ctx, nil); err != nil {
		return err
	}
	res.FinalPhase = *phase
	return nil
}

// DrainWorkflow empties a host of workloads before it is repaired or removed.
// Separate workflow because draining is driven by a different signal (health
// loss) and must be cancellable independently of provisioning.
func DrainWorkflow(ctx workflow.Context, hostID string) (fleet.Phase, error) {
	phase := fleet.PhaseReady
	opts := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
		StartToCloseTimeout: 15 * time.Minute,
		HeartbeatTimeout:    time.Minute,
		RetryPolicy:         &temporal.RetryPolicy{MaximumAttempts: 3},
	})

	next, err := fleet.Transition(phase, fleet.EventHealthLost)
	if err != nil {
		return phase, err
	}
	phase = next

	if err := workflow.ExecuteActivity(opts, DrainHost, hostID).Get(ctx, nil); err != nil {
		return phase, err
	}
	if next, err = fleet.Transition(phase, fleet.EventDrained); err != nil {
		return phase, err
	}
	return next, nil
}
