// The four-step progress timeline (approved mockup docs/design/m5-timeline-mockup.html).
// "list": the full steps while Becky works or after an error. "trail": the slim row above an answer.
import { stepViews, type Step } from "@/lib/timeline";

export default function Timeline({ steps, variant }: { steps: Step[]; variant: "list" | "trail" }) {
  return (
    <ol className={variant === "list" ? "timeline" : "trail"} aria-label={variant === "list" ? "Progress" : "Steps"}>
      {stepViews(steps, variant).map((v) => (
        <li key={v.label} className={`step-${v.status}`} aria-current={v.current ? "step" : undefined}>
          <span className={`dot ${v.status}`} aria-hidden="true">
            {v.icon}
          </span>
          <span>
            <span className="label">{v.label}</span>
            {v.meta && <span className="meta"> · {v.meta}</span>}
            {v.srText && <span className="sr-only"> ({v.srText})</span>}
          </span>
        </li>
      ))}
    </ol>
  );
}
