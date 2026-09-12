package workflow

import (
	"context"
	"fmt"
	"time"

	"go.temporal.io/sdk/activity"
)

// ValidationReport is the outcome of GPU burn-in.
type ValidationReport struct {
	Healthy     bool
	Reason      string
	GPUsFound   int
	ECCErrors   int
	NCCLBusGBps float64
}

// Provisioner is the side-effecting surface the activities call. It is an
// interface so the workflow can be tested against a fake, and so a second
// backend (a different BMC vendor, a different cloud) drops in without the
// workflow learning about it — the abstraction the platform exists to provide.
type Provisioner interface {
	WriteImage(ctx context.Context, hostID, image string) error
	InstallDrivers(ctx context.Context, hostID string) error
	Validate(ctx context.Context, hostID string, expectGPUs int, burnIn time.Duration) (ValidationReport, error)
	JoinPool(ctx context.Context, hostID string) error
	Release(ctx context.Context, hostID string) error
	Drain(ctx context.Context, hostID string) error
}

// Activities binds a Provisioner to the activity functions Temporal calls.
type Activities struct {
	P Provisioner
}

// WriteImage PXE-boots the host and writes the OS image. Heartbeats so a wedged
// write is caught by the heartbeat timeout instead of running to the 30-minute
// start-to-close timeout.
func WriteImage(ctx context.Context, req ProvisionRequest) error {
	a := fromContext(ctx)
	activity.RecordHeartbeat(ctx, "writing image")
	return a.P.WriteImage(ctx, req.HostID, req.Image)
}

// InstallDrivers installs the GPU driver and CUDA stack.
func InstallDrivers(ctx context.Context, req ProvisionRequest) error {
	a := fromContext(ctx)
	activity.RecordHeartbeat(ctx, "installing drivers")
	return a.P.InstallDrivers(ctx, req.HostID)
}

// ValidateGPUs runs burn-in: GPU count, ECC state, NCCL bus bandwidth.
//
// It returns a report rather than an error when the host is simply unhealthy.
// An unhealthy host is a normal outcome the workflow decides about; an error
// means the check itself could not run, and those are different things. Folding
// them together is how "validation failed" ends up retried four times by a
// retry policy that cannot tell the difference.
func ValidateGPUs(ctx context.Context, req ProvisionRequest) (ValidationReport, error) {
	a := fromContext(ctx)
	activity.RecordHeartbeat(ctx, "burn-in")
	burnIn := req.BurnInDuration
	if burnIn <= 0 {
		burnIn = 5 * time.Minute
	}
	rep, err := a.P.Validate(ctx, req.HostID, req.GPUs, burnIn)
	if err != nil {
		return ValidationReport{}, fmt.Errorf("burn-in could not run on %s: %w", req.HostID, err)
	}
	if rep.GPUsFound != req.GPUs {
		rep.Healthy = false
		rep.Reason = fmt.Sprintf("expected %d GPUs, found %d", req.GPUs, rep.GPUsFound)
	}
	return rep, nil
}

// JoinPool marks the host schedulable.
func JoinPool(ctx context.Context, req ProvisionRequest) error {
	return fromContext(ctx).P.JoinPool(ctx, req.HostID)
}

// ReleaseHost returns the host to the vendor pool or RMA.
func ReleaseHost(ctx context.Context, req ProvisionRequest) error {
	return fromContext(ctx).P.Release(ctx, req.HostID)
}

// DrainHost evicts workloads and marks the host unschedulable.
func DrainHost(ctx context.Context, hostID string) error {
	a := fromContext(ctx)
	activity.RecordHeartbeat(ctx, "draining")
	return a.P.Drain(ctx, hostID)
}

// activityKey carries the bound Activities through the activity context. A
// worker registers the struct's methods; these package-level functions read it
// back so the workflow refers to plain function names.
type activityKey struct{}

// WithActivities returns a context carrying the provisioner binding.
func WithActivities(ctx context.Context, a *Activities) context.Context {
	return context.WithValue(ctx, activityKey{}, a)
}

func fromContext(ctx context.Context) *Activities {
	if a, ok := ctx.Value(activityKey{}).(*Activities); ok && a != nil && a.P != nil {
		return a
	}
	panic("workflow: activities not bound to context; call WithActivities when starting the worker")
}
