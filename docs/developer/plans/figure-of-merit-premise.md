## Context

We need to explore an idea that might help to resolve a long standing problem: The experiment control software (NICOS) needs to read result streams of certain workflows to steer the experiment (e.g., to know when there are sufficient counts/statistics at a certain scan point and it can move to the next scan point). Not only does NICOS have to know which output of which workflow to use, but:
1. It needs to potentially configure the workflow so it can be started.
2. If it relies on workflows started by the dashboard, the JobID that is part of the result message ID is random and changes with every run.
Taken together, one would need to replicate signicant knowledge and logic that we have in the dashboard in NICOS.

## Idea

The core realization is really that in the instrument hutch NICOS and ESSlivedata would always run side-by-side. We should thus consider ESSlivedata "part" of the experiment control interface. Thus:

- Add a dedicated widget (or special options in existing widgets) that lets the user configure the *figure of merit* (is that a good name) that NICOS can use to steer the experiment run. This includes:
  - Which workflow.
  - Which output of the workflow.
  - Which parameters.
  - Start (or auto-start) that workflow.
- Backend should somehow (to be figured out how) publish the FOM under a statically known ID such that NICOS can always use this particular stream, without having to know what it is. Potentially on a dedicated new topic such that NICOS does not need to consume the full stream of all ESSlivedata jobs.
- NICOS will be responsible for *resetting* this job (as it might need to do so when moving to the next scan point).