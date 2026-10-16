export const HELP = `PLOINKY WORKERS — pworker

USAGE
  pworker                         Open the interactive menu.
  pworker DIRECTORY               Open the interactive menu in DIRECTORY.
  pworker --cwd DIRECTORY         Same as pworker DIRECTORY.
  pworker --help                  Show this help without starting the proxy.
  pworker run TASK [options]      Run a task file or compile a text request.
  pworker --async TASK [options]  Short form for run TASK --async.
  pworker --status [TASK_ID]      List detached tasks or inspect one task.

INTERACTIVE SETUP
  Log in / connect provider       Choose a built-in API provider and enter its
                                  key, add an OpenAI-compatible endpoint, or
                                  start a configured local model. The model
                                  catalog is checked before a connection is
                                  accepted.
  Configure tier                  Available only after a provider is connected.
                                  Choose a tier, then choose its primary model
                                  directly from one searchable live list across
                                  all connected providers. Each row shows the
                                  provider, price, and plan request cost. The
                                  prior usable entries remain fallbacks.
                                  Credit-balance and non-text models are hidden.
  Auto-configure price ladder     Uses published live prices to propose one
                                  model per tier from cheapest to most costly.
                                  First choose one provider, or explicitly
                                  choose to mix all providers. Review the
                                  proposal and confirm again before saving.
                                  Models known to require a credit balance and
                                  models without text output are excluded.
  Navigation                      Up/Down selects; Enter confirms; Esc cancels.
                                  Every menu also has Back / Cancel. Prompts
                                  can be canceled with Esc. A tier mapping is
                                  saved only after its final confirmation.

TASKS
  New task conversation           A text-mode conversation that starts every
                                  submitted request in the background. Keep
                                  adding requests while earlier work runs.
                                  The request is the task's initial input.
  Current tasks                   A separate monitor that lists at most the
                                  ten newest saved tasks, with state, current
                                  phase, and result access. Press Refresh to
                                  update task state.
  Interactive working directory   Tasks use the directory from which pworker
                                  was started. Use pworker DIRECTORY or
                                  pworker --cwd DIRECTORY to
                                  open the menu with another working directory.
  --purpose TAG                   Attach a purpose tag to request metrics (default pworker:cli).
  run TASK                        TASK is a .json or Markdown .md declaration (or static JSON
                                  .mjs declaration), natural-language
                                  request, or - to read the request from stdin.
                                  Only a direct JSON phase map executes.
                                  Functions, legacy task runners and executable
                                  task modules are not supported.
  --input TEXT_OR_JSON            Start-phase input. JSON objects become task
                                  variables; other values become \${input}.
  --cwd DIR                       Use the caller's exact directory for this
                                  task. Also available as
                                  --current-working-directory DIR. The phase
                                  receives this.currentWorkingDirectory and
                                  confined file operations.
  --async                         Return an ID without waiting for the task.
                                  Combine with run, --input, and --cwd.
  --status                        List all detached tasks, with phase, state,
                                  result, and error where available.
  --status TASK_ID                Inspect one ID returned by --async.

QUEUE AND BATCHING
  queue TASK [--input ...] [--cwd DIR]
                                  Add a task to the persistent queue without
                                  executing it. Repeat to accumulate work.
  queue list                      Show queued entries.
  queue clear                     Remove queued entries.
  flush                           Execute the queue and retain failed entries
                                  (exit code 1 when a task failed).
  flush --async                   Send the whole queue to one detached worker
                                  and return an ID for each task. Eligible
                                  phases can still share one model request.
  --batch                         With 'tier', enable prompt batching for that
                                  tier. Each task phase must also set batch:true
                                  and end its template with one \${variable}.

PROVIDERS AND TIERS
  provider NAME --endpoint URL [--key KEY] [--rpm N]
                                  Add an OpenAI-compatible provider. The live
                                  model catalog must respond before settings
                                  are saved. --key is the provider API key;
                                  omit it only for a keyless local endpoint.
                                  --rpm is requests per minute (default 60).
                                  Passing a key on the command line may leave
                                  it in shell history; the interactive menu
                                  masks key entry.
  tier TIER --provider NAME --model MODEL_ID [--add] [--batch]
                                  Assign a connected provider and a model from
                                  its live catalog to a tier. --add retains
                                  the existing entries and appends this one as
                                  a fallback. Without --add, it replaces the
                                  tier. Models known to require a credit
                                  balance or without text output are rejected.
  models                          Show configured local models and status.
  models --provider NAME          List the live remote model catalog.
  models --all                    List connected catalogs, reporting failures.
  models start NAME               Start a configured local model server.
  models stop NAME                Stop a configured local model server.

LOCAL-LLM ENGINE (models the proxy runs itself: on Linux, and on macOS with
                  Metal on Apple Silicon or the CPU build on an Intel Mac)
  models catalog | runners [--json]
                                  The models the engine can run, and its
                                  runners (the ones this machine cannot use
                                  are listed with the reason).
  models runner install ID [--accept-licence] [--detach]
  models runner uninstall ID      Install or remove a runner (for example
                                  llama.cpp-cpu). Progress is shown.
  models preview ID [--runner R] [--ctx N] [--parallel N] [--param k=v]
                                  The admission verdict, no process started.
  models run ID [--runner R] [--ctx N] [--parallel N] [--param k=v]
                [--replace] [--detach]
                                  Download (first time) and start the one
                                  deployment. Shows the phase, bytes, rate and
                                  ETA until it is ready (exit 0), fails (1) or
                                  is interrupted (130, the download goes on).
  models stop                     Stop the deployment (with a name: a local
                                  model server, as above).
  models cancel                   Cancel its download.
  models status [--follow] [--json]
                                  Phase and deployment; says why the engine is
                                  unavailable (disabled, owned by another
                                  process, unsupported platform).
  models delete-weights ID [--runner R]
                                  Delete the downloaded weights.
  models lookup REPO [--file F] [--revision R]
  models add FILE|-  |  update FILE|-  |  remove ID
                                  Look a Hugging Face model up, or manage your
                                  own models (a JSON catalog entry).
  models hf-token                 Store a Hugging Face token in
                                  keys/huggingface.env (mode 0600). The token
                                  is read with hidden input, or from stdin when
                                  it is not a terminal, and never printed.
                                  Restart the proxy (stop, start) to use it.

PROXY AND MONITORING
  start                           Start the independent proxy process.
  stop                            Stop that proxy process.
  serve [--port N] [--host HOST]  Run the proxy in the foreground. Default:
                                  127.0.0.1:18080.
  stats [--since 1h|24h|7d] [--provider NAME] [--by hour|day] [--json]
                                  Request metrics from the request log (no
                                  running proxy needed): per provider/model
                                  and per hour or day, attempts sent, ok,
                                  429s, 429s while within the configured
                                  rates (429<lim), errors, retries, timeouts,
                                  cache hits, requests saved by batching,
                                  tokens, queue wait, peak and average rate.
  stats --proxy                   The running proxy's raw /stats JSON.

EXAMPLES
  pworker
  pworker run ./task.json --input '{"input":"hello"}' --cwd ./project
  pworker run 'Summarize the supplied text' --input 'Meeting notes...' --async
  pworker --status TASK_ID
  pworker queue ./task.json --input first --cwd ./project
  pworker queue ./task.json --input second --cwd ./project
  pworker flush --async
  pworker provider myapi --endpoint https://example.com/v1 --key KEY --rpm 30
  pworker tier small --provider myapi --model MODEL_ID --batch
  pworker tier small --provider backup --model BACKUP_MODEL --add

DATA AND ENVIRONMENT
  PWORKER_HOME                   User config, keys, tasks, jobs, logs, and cache
                                  directory (default ~/.pworker).
  PWORKER_PORT                   Local proxy port when no PWORKER_URL is set.
  PWORKER_URL                    Override the proxy URL for the client.
  PWORKER_CONFIG                 Optional project configuration layer.
  PWORKER_TOKEN                  Optional proxy access token.

Documentation: PWORKER.md
`;
