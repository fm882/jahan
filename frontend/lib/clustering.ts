import type { ClusteringCoverage } from "./engine";

export function clusteringNotice(coverage: ClusteringCoverage[] | undefined, noun: string): string {
  if (coverage === undefined) return "Clustering coverage was not recorded in this report.";
  return coverage.map((c) => {
    const notice = `${c.excluded} of ${c.eligible} ${noun} could not be grouped.`;
    return c.eligible > 0 && c.clustered === 0 ? `Clustering unavailable. ${notice}` : notice;
  }).join(" ");
}
