# Superseded — do not report from this folder

Second scoring (after the review fixes). Claude's own check found that the new per-stage rule
("value > 0") also dropped the agent's evidence-gate stage, which runs on every question but takes
under 0.05 ms and is stored as 0.0, so it showed n=0. Fixed: stages after the gate count only when the
gate passed; all other stages count on every question. Only per-stage latency figures differ; every
other summary value and the held claim-support file are identical to the first scoring (checked by
diff / hash). Kept on disk as the record.
