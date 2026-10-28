// Becky, the original mascot (owner-approved mockup: docs/design/m5-home-mockup.html).
// Three poses: "normal" (stack of books), "working" (reading, page flip), "noAnswer" (apologetic shrug).
import { useId } from "react";

export type BeckyPose = "normal" | "working" | "noAnswer";

const LABELS: Record<BeckyPose, string> = {
  normal: "Becky, a bunny in a lab coat, holding a stack of books",
  working: "Becky reading an open book",
  noAnswer: "Becky shrugging with an apologetic smile",
};

const NAVY = "#1E2A47";

export default function Becky({ pose, width = 230 }: { pose: BeckyPose; width?: number }) {
  // Gradient ids must be unique per SVG on the page.
  const p = useId().replace(/:/g, "");
  const u = (name: string) => `url(#${p}-${name})`;
  const sleeve = (d: string) => (
    <>
      <path d={d} fill="none" stroke="#C9D3E5" strokeWidth={22} strokeLinecap="round" strokeLinejoin="round" transform="translate(1.5 2)" />
      <path d={d} fill="none" stroke="#F6F8FC" strokeWidth={20} strokeLinecap="round" strokeLinejoin="round" />
    </>
  );
  const paw = (cx: number, cy: number) => <circle cx={cx} cy={cy} r={10} fill={u("fur")} />;

  const ears = (
    <>
      <ellipse cx="84" cy="50" rx="20" ry="44" transform="rotate(-14 84 50)" fill={u("ear")} />
      <ellipse cx="85" cy="55" rx="10" ry="31" transform="rotate(-14 85 55)" fill={u("innerEar")} />
      <ellipse cx="136" cy="50" rx="20" ry="44" transform="rotate(14 136 50)" fill={u("ear")} />
      <ellipse cx="135" cy="55" rx="10" ry="31" transform="rotate(14 135 55)" fill={u("innerEar")} />
    </>
  );
  const body = (
    <>
      <ellipse cx="88" cy="223" rx="21" ry="11" fill={u("fur")} />
      <ellipse cx="132" cy="223" rx="21" ry="11" fill={u("fur")} />
      <path d="M76 132 C61 148 56 180 58 206 C59 218 66 224 78 224 L142 224 C154 224 161 218 162 206 C164 180 159 148 144 132 Z" fill={u("coat")} />
      <path d="M96 132 L110 156 L124 132 Z" fill="#A9C6FF" />
      <path d="M93 134 L110 172 L127 134" fill="none" stroke="#C3CEE2" strokeWidth={3.5} strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="110" cy="188" r="3.2" fill="#C3CEE2" />
      <circle cx="110" cy="204" r="3.2" fill="#C3CEE2" />
      <rect x="133" y="167" width="5" height="13" rx="2.5" fill="#4F86E8" />
      <rect x="128" y="175" width="21" height="16" rx="5" fill="#F4F7FC" stroke="#D2DBEA" strokeWidth={2} />
    </>
  );
  const head = (
    <>
      <ellipse cx="110" cy="96" rx="60" ry="51" fill={u("fur")} />
      <ellipse cx="86" cy="68" rx="24" ry="12" transform="rotate(-22 86 68)" fill="#FFFFFF" opacity={0.38} />
      <ellipse cx="80" cy="114" rx="13" ry="8" fill={u("blush")} />
      <ellipse cx="140" cy="114" rx="13" ry="8" fill={u("blush")} />
      <ellipse cx="110" cy="115" rx="17" ry="12" fill="#FFF3E4" opacity={0.9} />
      <path d="M103 107 Q110 103 117 107 Q114 114 110 115 Q106 114 103 107 Z" fill="#EE8A88" />
      <ellipse cx="108" cy="107.5" rx="2.6" ry="1.4" fill="#FFFFFF" opacity={0.7} />
      {/* hair bow */}
      <ellipse cx="129" cy="51" rx="10" ry="7.5" transform="rotate(-25 129 51)" fill={u("bow")} />
      <ellipse cx="148" cy="51" rx="10" ry="7.5" transform="rotate(25 148 51)" fill={u("bow")} />
      <circle cx="138.5" cy="54" r="5" fill="#3A6CC6" />
      <ellipse cx="126" cy="48" rx="3.5" ry="2" fill="#FFFFFF" opacity={0.55} />
    </>
  );
  const openEyes = (
    <>
      <ellipse cx="88" cy="94" rx="11.5" ry="13.5" fill={u("eye")} />
      <circle cx="92.5" cy="88.5" r="4.6" fill="#fff" />
      <circle cx="84.5" cy="99.5" r="2" fill="#fff" opacity={0.85} />
      <ellipse cx="132" cy="94" rx="11.5" ry="13.5" fill={u("eye")} />
      <circle cx="136.5" cy="88.5" r="4.6" fill="#fff" />
      <circle cx="128.5" cy="99.5" r="2" fill="#fff" opacity={0.85} />
      <path d="M77 86 l-6 -4 M77.5 91 l-7 -1 M143 86 l6 -4 M142.5 91 l7 -1" stroke={NAVY} strokeWidth={2.6} strokeLinecap="round" />
    </>
  );
  const smile = (
    <path d="M110 115 Q110 121 103 121 M110 115 Q110 121 117 121" fill="none" stroke={NAVY} strokeWidth={2.6} strokeLinecap="round" />
  );

  return (
    <svg width={width} height={Math.round((width * 242) / 220)} viewBox="0 0 220 242" role="img" aria-label={LABELS[pose]} className="becky">
      <defs>
        <radialGradient id={`${p}-fur`} cx="35%" cy="28%" r="80%">
          <stop offset="0" stopColor="#FFF3E2" />
          <stop offset=".55" stopColor="#F4D5AF" />
          <stop offset="1" stopColor="#DCAE84" />
        </radialGradient>
        <linearGradient id={`${p}-ear`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#FCE6CB" />
          <stop offset="1" stopColor="#E2B78E" />
        </linearGradient>
        <linearGradient id={`${p}-innerEar`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#FBC4B4" />
          <stop offset="1" stopColor="#EE9C8A" />
        </linearGradient>
        <radialGradient id={`${p}-coat`} cx="35%" cy="20%" r="90%">
          <stop offset="0" stopColor="#FFFFFF" />
          <stop offset=".6" stopColor="#F3F6FB" />
          <stop offset="1" stopColor="#D6DFEE" />
        </radialGradient>
        <radialGradient id={`${p}-eye`} cx="40%" cy="35%" r="70%">
          <stop offset="0" stopColor="#3E4F7A" />
          <stop offset="1" stopColor="#111A30" />
        </radialGradient>
        <radialGradient id={`${p}-blush`} cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="#F59A92" stopOpacity={0.85} />
          <stop offset="1" stopColor="#F59A92" stopOpacity={0} />
        </radialGradient>
        <radialGradient id={`${p}-floor`} cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor={NAVY} stopOpacity={0.22} />
          <stop offset="1" stopColor={NAVY} stopOpacity={0} />
        </radialGradient>
        <linearGradient id={`${p}-bow`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#7FA8F5" />
          <stop offset="1" stopColor="#3A6CC6" />
        </linearGradient>
        <linearGradient id={`${p}-bookBlue`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#78A2F3" />
          <stop offset="1" stopColor="#3A6CC6" />
        </linearGradient>
        <linearGradient id={`${p}-bookMint`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#93DDBC" />
          <stop offset="1" stopColor="#49A882" />
        </linearGradient>
        <linearGradient id={`${p}-bookPeach`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#FFC5AD" />
          <stop offset="1" stopColor="#E58C69" />
        </linearGradient>
        <linearGradient id={`${p}-page`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#EDE5D8" />
          <stop offset=".35" stopColor="#FFFFFF" />
          <stop offset="1" stopColor="#F7F2EA" />
        </linearGradient>
        <linearGradient id={`${p}-pageR`} x1="1" y1="0" x2="0" y2="0">
          <stop offset="0" stopColor="#F7F2EA" />
          <stop offset=".65" stopColor="#FFFFFF" />
          <stop offset="1" stopColor="#EDE5D8" />
        </linearGradient>
      </defs>

      <ellipse className="becky-floor" cx="110" cy="232" rx="62" ry="9" fill={u("floor")} />

      {pose === "normal" && (
        <g className="becky-bob">
          {ears}
          {body}
          {sleeve("M80 148 L80 186")}
          {sleeve("M140 148 L140 186")}
          <rect x="80" y="191" width="62" height="15" rx="4" fill={u("bookBlue")} />
          <rect x="134" y="193" width="6" height="11" rx="1.5" fill="#FBF6EE" />
          <g transform="rotate(-4 110 184)">
            <rect x="84" y="177" width="56" height="14" rx="4" fill={u("bookMint")} />
            <rect x="86" y="179" width="6" height="10" rx="1.5" fill="#FBF6EE" />
          </g>
          <g transform="rotate(3 110 170)">
            <rect x="82" y="163" width="58" height="14" rx="4" fill={u("bookPeach")} />
            <rect x="132" y="165" width="6" height="10" rx="1.5" fill="#FBF6EE" />
            <rect x="96" y="167.5" width="22" height="3" rx="1.5" fill="#FFFFFF" opacity={0.55} />
          </g>
          {paw(79, 190)}
          {paw(141, 190)}
          {head}
          {openEyes}
          {smile}
        </g>
      )}

      {pose === "working" && (
        <g className="becky-read">
          {ears}
          {body}
          {head}
          <g transform="translate(0 5)">{openEyes}</g>
          <path d="M104 119 Q110 123 116 119" fill="none" stroke={NAVY} strokeWidth={2.6} strokeLinecap="round" />
          {sleeve("M80 150 L72 172")}
          {sleeve("M140 150 L148 172")}
          <path d="M110 190 Q90 180 64 184 L64 150 Q90 146 110 156 Q130 146 156 150 L156 184 Q130 180 110 190 Z" fill={u("bookBlue")} />
          <path d="M110 185 Q91 175 68 179 L68 147 Q91 143 110 153 Z" fill={u("page")} />
          <path d="M110 185 Q129 175 152 179 L152 147 Q129 143 110 153 Z" fill={u("pageR")} />
          <path d="M76 155 Q90 152 102 158 M76 162 Q90 159 102 165 M76 169 Q88 166 98 171 M118 158 Q130 152 144 155 M118 165 Q130 159 144 162 M122 171 Q132 166 144 169" fill="none" stroke="#C9C1B3" strokeWidth={2} strokeLinecap="round" />
          {/* turning page: flips right to left around the spine (x = 110) */}
          <path className="becky-flip" d="M110 185 Q129 175 152 179 L152 147 Q129 143 110 153 Z" fill={u("pageR")} stroke="#E4DACB" strokeWidth={1} />
          <path d="M110 153 L110 185" stroke="#D8CDBB" strokeWidth={2} />
          {paw(68, 172)}
          {paw(152, 172)}
        </g>
      )}

      {pose === "noAnswer" && (
        <g className="becky-bob">
          {body}
          {sleeve("M78 148 L56 166 L44 150")}
          {sleeve("M142 148 L164 166 L176 150")}
          {paw(42, 144)}
          {paw(178, 144)}
          <g transform="rotate(-7 110 140)">
            {ears}
            {head}
            {/* smiling closed eyes, brows raised in the middle: "sorry!" rather than sad */}
            <path d="M78 96 Q88 86 98 96 M122 96 Q132 86 142 96" fill="none" stroke={NAVY} strokeWidth={4} strokeLinecap="round" />
            <path d="M76 76 Q86 74 96 68 M124 68 Q134 74 144 76" fill="none" stroke="#B8875E" strokeWidth={3.5} strokeLinecap="round" />
            <path d="M102 119 Q110 125 118 119" fill="none" stroke={NAVY} strokeWidth={2.8} strokeLinecap="round" />
          </g>
        </g>
      )}
    </svg>
  );
}
