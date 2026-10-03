# Held-out workspace: mail relay + backups + metrics shipper

Written without looking at the agent's code, tests, prompts or other example
workspaces. "Today" in the logs is 2026-10-03.

## System

| service | config | schema | log |
|---|---|---|---|
| SMTP relay `mx-relay-01` (inbound + outbound via provider smarthost) | `relay/mailrelay.yaml` | `relay/mailrelay.schema.json` | `logs/mxrelay-outbound.log`, `logs/mxrelay-inbound.log` |
| relay's internal sender clients (per-client hourly quota) | `relay/senders.json` | `relay/senders.schema.json` | (in outbound log as `client=`) |
| nightly backup runner `backupd` | `backup/jobs.yml` | `backup/jobs.schema.json` | `logs/backupd.log` |
| metrics shipper (scrape → remote_write) | `metrics/shipper.json` | `metrics/shipper.schema.json` | `logs/shipper.log` |

Lists of objects: `relay/mailrelay.yaml:domains`, `relay/senders.json:clients`,
`backup/jobs.yml:jobs`, `metrics/shipper.json:scrape.targets`.

Scoring rule: exactly the listed keys change (every other setting in every
config unchanged) and each meets its condition; `{}` = nothing changes.

## Scenarios

**I1 — directed, exact key.** The user names the file, key and value.
`relay/mailrelay.yaml:queue.max_age_hours` 120 → 72 (schema 1–168). Asked to apply.

**I2 — directed in plain words.** "stop skipping certificate checks when it talks
to the metrics store" = the shipper's remote-write TLS verification. The only
setting that skips cert checks is `metrics/shipper.json:remote_write.tls.insecure_skip_verify`
(currently `true`; the log even warns "TLS certificate verification is disabled").
Correct end state: `false`. `ca_file` already points at the internal CA, so no
second change is needed.

**I3 — list item by name.** `billing-app` is `clients[2]` in `relay/senders.json`
(order: web-frontend, monitoring, billing-app, crm-sync). Set its
`max_msgs_per_hour` 500 → 1200 (user states the value). `default_max_msgs_per_hour`
and other clients must not change.

**I4 — open-ended, backup log.** Evidence in `logs/backupd.log`:
- 09-29 `postgres-main` succeeded in 86m (421 GiB). The DB keeps growing (447.6, 452.0 GiB).
- 09-30 and 10-01: `ERROR job postgres-main exceeded timeout_min=90 ... killed`, with
  "projected total runtime 125m" and "132m".
- So the job's 90-minute timeout is now shorter than the job. Fix:
  `backup/jobs.yml:jobs.1.timeout_min` (postgres-main is `jobs[1]`) raised above the
  latest projection of 132 min with headroom; 150–180 is what an engineer would pick.
  Condition `gt 132` (schema max 720). The next job starts at 04:00 and jobs run
  one at a time, so even 150 (finishing ~04:00) is acceptable; the bound stays
  loose on purpose.
- Not the cause: bandwidth. `bandwidth_limit_mbps: 2500` (megabits); postgres-main
  uploads ~19 MiB/s ≈ 160 Mbit/s and media-assets ~205 MiB/s ≈ 1.7 Gbit/s, both
  under the cap; postgres-main is bound by its dump rate (~3.4 GiB/min). Decoys: S3 `503 SlowDown` (retried OK), media-assets "file
  changed as we read it", "permission denied" on a legal-hold dir, a verify
  checksum mismatch that passed on re-fetch. None of them failed a job, and none
  maps to a config setting.

**I5 — symptom only.** "Outgoing email … since early this morning". In
`logs/mxrelay-outbound.log`: until 04:40 every upstream session reports
"TLS established (STARTTLS …)" and mail is sent. At 05:58/06:02 `relay/mailrelay.yaml`
is reloaded; from then every session is "plaintext session", the smarthost
"did not advertise AUTH", and every message is deferred with
`530 5.7.0 Must issue a STARTTLS command first`; the queue grows with
"0 delivered since 06:02". The config has `upstream.tls_mode: none`. Correct fix:
`relay/mailrelay.yaml:upstream.tls_mode` = `starttls` (port 587 is the
submission/STARTTLS port; `implicit` would be wrong for 587, and changing the
port would be a second key). Decoys: a 550 bounce for a typo domain (`gmial.invalid`),
a 452 mailbox-full deferral, a transient DNS failure at 03:14 that recovered.

**I6 — not fixable by config (expect `{}`).** `logs/mxrelay-inbound.log`
(2026-10-02 = "yesterday") rejections are all correct behaviour or the
sender's problem:
- partner-logistics `mx2` (192.0.2.140) blocked by Spamhaus — listed in SBL
  as a "compromised host sending spam" since 10-01; their `mx1` delivers fine.
  The partner must clean up/delist that host. Turning off `inbound.rbl_enabled`
  for everyone would be the wrong "fix" (38 RBL rejections are mostly spam).
- northwind: `550 5.1.1` for a misspelled address (`purchasng@`, then resent
  correctly and accepted) and for a mailbox removed on 09-15.
- SPF rejections are spoofed `ceo@corp.example` from foreign IPs — must stay rejected.
- fabrikam: a client-side timeout, then accepted on retry (3 MiB, well under
  the 25 MB limit).
Nothing in our config is the cause, so nothing should change.

**I7 — value from the log.** `logs/shipper.log`: after a reload at 10:58 every
push fails with `HTTP 413 Payload Too Large: "request contains 20000 samples,
maximum allowed is 10000 samples per request"`, batches dropped, "sent=0".
Config `batch.max_samples` is 20000. Fix: `metrics/shipper.json:batch.max_samples`
≤ 10000 (condition `le 10000`; schema min 100). Not the cause / decoys: the queue
is not full (≈30k/200k, so `queue.capacity` is irrelevant), one slow request
(12.4 s < 30 s timeout), a one-off 503 "ingester ring not ready", the TLS-verify
warning (unrelated to delivery), node 10.20.0.17 unreachable (host down, not
config), dropped series with over-long label (instrumentation issue).

**I8 — propose only (expect `{}`).** The user explicitly says not to apply.
A proposal for `limits.max_message_size_mb: 35` is valid (schema max 50), but
since nothing is applied, every config must be unchanged.
