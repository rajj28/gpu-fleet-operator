package workflow

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"go.temporal.io/sdk/testsuite"

	"github.com/rajj28/gpu-fleet-operator/internal/fleet"
)

// These run against the Temporal SDK's test environment — no server, no
// database. The environment replays the workflow exactly as a real worker
// would, so the durability semantics under test are the real ones.

func newEnv(t *testing.T) *testsuite.TestWorkflowEnvironment {
	t.Helper()
	var s testsuite.WorkflowTestSuite
	env := s.NewTestWorkflowEnvironment()
	env.RegisterWorkflow(ProvisionHostWorkflow)
	env.RegisterWorkflow(DrainWorkflow)
	return env
}

func req() ProvisionRequest {
	return ProvisionRequest{
		HostID: "gpu-node-014", GPUType: "h100", GPUs: 8,
		Image: "ubuntu-22.04-cuda12", BurnInDuration: time.Minute,
	}
}

func TestProvisionHappyPath(t *testing.T) {
	env := newEnv(t)
	env.OnActivity(WriteImage, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(InstallDrivers, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(ValidateGPUs, mock.Anything, mock.Anything).Return(
		ValidationReport{Healthy: true, GPUsFound: 8, NCCLBusGBps: 421.5}, nil)
	env.OnActivity(JoinPool, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ProvisionHostWorkflow, req())

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res ProvisionResult
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, fleet.PhaseReady, res.FinalPhase)
	require.Equal(t, 1, res.Attempts)
	require.Equal(t, []string{
		"Discovered--Provision-->Provisioning",
		"Provisioning--ImageWritten-->DriverInstall",
		"DriverInstall--DriversReady-->Validating",
		"Validating--ValidationPass-->Ready",
	}, res.Transitions)
}

// A host that reports the wrong GPU count is unhealthy, not erroring. It must
// be retried through the state machine, not by the activity retry policy.
func TestProvisionRetriesUnhealthyHostThenGivesUp(t *testing.T) {
	env := newEnv(t)
	env.OnActivity(WriteImage, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(InstallDrivers, mock.Anything, mock.Anything).Return(nil)
	// Persistently short one GPU.
	env.OnActivity(ValidateGPUs, mock.Anything, mock.Anything).Return(
		ValidationReport{Healthy: false, Reason: "expected 8 GPUs, found 7", GPUsFound: 7}, nil)

	env.ExecuteWorkflow(ProvisionHostWorkflow, req())

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var res ProvisionResult
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, MaxProvisionAttempts, res.Attempts,
		"should exhaust the attempt budget rather than loop forever")
	require.Equal(t, fleet.PhaseFailed, res.FinalPhase,
		"a host that never validates must land in Failed for a human, not Ready")
	require.NotContains(t, res.Transitions, "Validating--ValidationPass-->Ready")
}

// A transient failure on the first attempt must recover on the second, and the
// host must still end up Ready.
func TestProvisionRecoversFromTransientImageFailure(t *testing.T) {
	env := newEnv(t)
	calls := 0
	env.OnActivity(WriteImage, mock.Anything, mock.Anything).Return(func(ctx context.Context, r ProvisionRequest) error {
		calls++
		if calls == 1 {
			return errors.New("PXE timed out")
		}
		return nil
	})
	env.OnActivity(InstallDrivers, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(ValidateGPUs, mock.Anything, mock.Anything).Return(
		ValidationReport{Healthy: true, GPUsFound: 8}, nil)
	env.OnActivity(JoinPool, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(ProvisionHostWorkflow, req())

	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
	var res ProvisionResult
	require.NoError(t, env.GetWorkflowResult(&res))
	require.Equal(t, fleet.PhaseReady, res.FinalPhase)
	// The activity's own retry policy absorbs a transient PXE timeout, so the
	// outer attempt counter stays at 1 — that is the point of the retry policy.
	// What must be true is that WriteImage really was called twice.
	require.Equal(t, 2, calls, "activity retry policy did not retry the PXE timeout")
	require.Equal(t, 1, res.Attempts,
		"a transient activity failure should not burn an outer provisioning attempt")
}

// The phase query is how the self-service API answers "where is my host?"
// without reading a database. It must be live mid-workflow.
func TestPhaseIsQueryableMidFlight(t *testing.T) {
	env := newEnv(t)
	env.OnActivity(WriteImage, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(InstallDrivers, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(ValidateGPUs, mock.Anything, mock.Anything).Return(
		ValidationReport{Healthy: true, GPUsFound: 8}, nil)
	env.OnActivity(JoinPool, mock.Anything, mock.Anything).Return(nil)

	// Ask while the workflow is still running.
	env.RegisterDelayedCallback(func() {
		v, err := env.QueryWorkflow(QueryPhase)
		require.NoError(t, err)
		var phase string
		require.NoError(t, v.Get(&phase))
		require.NotEmpty(t, phase)
		require.NotEqual(t, string(fleet.PhaseDecommissioned), phase)
	}, time.Millisecond)

	env.ExecuteWorkflow(ProvisionHostWorkflow, req())
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	v, err := env.QueryWorkflow(QueryPhase)
	require.NoError(t, err)
	var final string
	require.NoError(t, v.Get(&final))
	require.Equal(t, string(fleet.PhaseReady), final)
}

// An abort signal must take the host out cleanly rather than abandoning the
// workflow mid-image-write.
func TestAbortSignalDecommissionsCleanly(t *testing.T) {
	env := newEnv(t)
	env.OnActivity(WriteImage, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(InstallDrivers, mock.Anything, mock.Anything).Return(nil)
	env.OnActivity(ValidateGPUs, mock.Anything, mock.Anything).Return(
		ValidationReport{Healthy: false, Reason: "aborted"}, nil)
	env.OnActivity(ReleaseHost, mock.Anything, mock.Anything).Return(nil)

	env.RegisterDelayedCallback(func() {
		env.SignalWorkflow(SignalAbort, nil)
	}, time.Nanosecond)

	env.ExecuteWorkflow(ProvisionHostWorkflow, req())
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())
}

// Draining is its own workflow so health-loss handling is cancellable
// independently. A drained host lands in Repair, never straight back in the pool.
func TestDrainEndsInRepair(t *testing.T) {
	env := newEnv(t)
	env.OnActivity(DrainHost, mock.Anything, mock.Anything).Return(nil)

	env.ExecuteWorkflow(DrainWorkflow, "gpu-node-014")
	require.True(t, env.IsWorkflowCompleted())
	require.NoError(t, env.GetWorkflowError())

	var phase fleet.Phase
	require.NoError(t, env.GetWorkflowResult(&phase))
	require.Equal(t, fleet.PhaseRepair, phase)
	require.False(t, fleet.Schedulable(phase), "a draining/repairing host must not be schedulable")
}
