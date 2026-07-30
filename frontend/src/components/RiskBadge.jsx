const STYLES = {
  Low: "bg-green-100 text-green-800 border-green-300",
  Medium: "bg-amber-100 text-amber-800 border-amber-300",
  High: "bg-red-100 text-red-800 border-red-300",
};

const EMOJI = { Low: "🟢", Medium: "🟡", High: "🔴" };

export default function RiskBadge({ level }) {
  const style = STYLES[level] || STYLES.Medium;
  return (
    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium border ${style}`}>
      <span>{EMOJI[level] || "🟡"}</span> {level} Risk
    </span>
  );
}
