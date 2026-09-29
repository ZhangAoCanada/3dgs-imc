export const meta = {
  name: 'curate-vision-crops',
  description: 'Vision-curate candidate comparison crops: rank by readability + obviousness of Ours win',
  phases: [
    { title: 'Judge', detail: 'one vision agent per (stage,scene) ranks its candidate strips' },
    { title: 'Verify', detail: 'adversarial recheck of the top picks (is the win really obvious?)' },
  ],
}

// groups inlined (args delivery is unreliable in this harness)
const GROUPS = [{"stage": "coverage_gap", "scene": "train_resize", "dir": "/tmp/cand_tnt_cg/train_resize", "ids": ["t000", "t001", "t002", "t003", "t004", "t005", "t006", "t007", "t008", "t009", "t010", "t011", "t012", "t013", "t014", "t015"]}, {"stage": "coverage_gap", "scene": "train_resize", "dir": "/tmp/cand_tnt_cg/train_resize", "ids": ["t016", "t017"]}, {"stage": "coverage_gap", "scene": "truck_resize", "dir": "/tmp/cand_tnt_cg/truck_resize", "ids": ["t018", "t019", "t020", "t021", "t022", "t023", "t024", "t025", "t026", "t027", "t028", "t029", "t030", "t031", "t032", "t033"]}, {"stage": "coverage_gap", "scene": "truck_resize", "dir": "/tmp/cand_tnt_cg/truck_resize", "ids": ["t034", "t035"]}, {"stage": "discontinuous", "scene": "building-small", "dir": "/tmp/cand_tnt_disc/building-small", "ids": ["t000", "t001", "t002", "t003", "t004", "t005", "t006"]}, {"stage": "discontinuous", "scene": "rubble-small", "dir": "/tmp/cand_tnt_disc/rubble-small", "ids": ["t007", "t008", "t009", "t010", "t011", "t012", "t013", "t014", "t015", "t016", "t017", "t018", "t019", "t020", "t021", "t022"]}, {"stage": "discontinuous", "scene": "rubble-small", "dir": "/tmp/cand_tnt_disc/rubble-small", "ids": ["t023", "t024", "t025", "t026", "t027", "t028", "t029", "t030", "t031", "t032", "t033", "t034", "t035", "t036", "t037", "t038"]}, {"stage": "discontinuous", "scene": "rubble-small", "dir": "/tmp/cand_tnt_disc/rubble-small", "ids": ["t039", "t040", "t041", "t042"]}, {"stage": "discontinuous", "scene": "train_resize", "dir": "/tmp/cand_tnt_disc/train_resize", "ids": ["t043", "t044", "t045", "t046", "t047", "t048", "t049", "t050", "t051", "t052", "t053", "t054", "t055", "t056", "t057", "t058"]}, {"stage": "discontinuous", "scene": "train_resize", "dir": "/tmp/cand_tnt_disc/train_resize", "ids": ["t059", "t060", "t061", "t062", "t063", "t064", "t065", "t066", "t067", "t068", "t069", "t070", "t071", "t072", "t073", "t074"]}, {"stage": "discontinuous", "scene": "train_resize", "dir": "/tmp/cand_tnt_disc/train_resize", "ids": ["t075", "t076", "t077", "t078"]}, {"stage": "discontinuous", "scene": "truck_resize", "dir": "/tmp/cand_tnt_disc/truck_resize", "ids": ["t079", "t080", "t081", "t082", "t083", "t084", "t085", "t086", "t087", "t088", "t089", "t090", "t091", "t092", "t093", "t094"]}, {"stage": "discontinuous", "scene": "truck_resize", "dir": "/tmp/cand_tnt_disc/truck_resize", "ids": ["t095", "t096", "t097", "t098", "t099", "t100", "t101", "t102", "t103", "t104", "t105", "t106", "t107", "t108", "t109", "t110"]}, {"stage": "discontinuous", "scene": "truck_resize", "dir": "/tmp/cand_tnt_disc/truck_resize", "ids": ["t111", "t112", "t113", "t114"]}]

const RANK_SCHEMA = {
  type: 'object',
  required: ['ranked'],
  properties: {
    ranked: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'readable', 'obvious_win', 'recommend'],
        properties: {
          id: { type: 'string', description: 'candidate id, e.g. c012' },
          readable: { type: 'integer', description: '0=flat sky/wall/shadow or pure mush, no recognizable structure; 3=clear recognizable object/edges/contours' },
          obvious_win: { type: 'integer', description: '0=Ours looks same as baselines; 3=Ours obviously closer to GT than every baseline, a layperson would see it' },
          artifact: { type: 'string', description: 'dominant baseline failure: ghost|doubling|blur|floater|missing-geometry|color-shift|none' },
          worst_baseline: { type: 'string' },
          caption: { type: 'string', description: 'one concrete sentence: what structure is shown + how baselines fail + how Ours is correct' },
          recommend: { type: 'boolean', description: 'true ONLY if readable>=2 AND obvious_win>=2' },
        },
      },
    },
  },
}

const VERDICT_SCHEMA = {
  type: 'object',
  required: ['confirm', 'readable', 'obvious_win'],
  properties: {
    confirm: { type: 'boolean', description: 'true if this crop is BOTH readable (recognizable structure, not flat/mush) AND shows an obvious Ours advantage' },
    readable: { type: 'integer' },
    obvious_win: { type: 'integer' },
    artifact: { type: 'string' },
    caption: { type: 'string' },
    reason: { type: 'string', description: 'why confirmed or rejected (be skeptical; reject flat sky/wall/shadow and look-alike crops)' },
  },
}

function judgePrompt(g) {
  const lines = g.ids.map(id => `- ${id}: ${g.dir}/${id}.png`).join('\n')
  return `You are curating qualitative comparison crops for a 3D Gaussian Splatting paper figure ` +
    `(${g.stage} setting, scene "${g.scene}"). Read EVERY candidate image listed below with the Read tool.\n\n` +
    `Each image is ONE candidate laid out as: [full ground-truth frame with a RED BOX marking the crop region] | ` +
    `[GT crop] | [Ours crop, GREEN border] | [baseline crops]. "Ours" should look closest to the GT crop; ` +
    `baselines should be worse.\n\n` +
    `Rate each candidate on two axes (integers 0-3):\n` +
    `  • readable: can a viewer instantly tell WHAT they are looking at? 3 = clear recognizable structure ` +
    `(object edges, contours, architecture, furniture, text, patterns). 0 = flat sky, flat wall/ceiling, pure ` +
    `shadow, or undifferentiated foliage mush with NO semantic anchor. Give 0-1 to anything mostly white, mostly ` +
    `black, or a featureless gradient.\n` +
    `  • obvious_win: is "Ours" VISIBLY closer to the GT crop than the baselines, in a way a non-expert would ` +
    `immediately point to? 3 = a baseline shows a blatant ghost / double-edge / blur / floater / missing geometry ` +
    `right where Ours is sharp and correct. 0 = Ours and the baselines look basically the same.\n\n` +
    `Be STRICT and adversarial — we need crops where the difference JUMPS OUT and the content is clearly readable. ` +
    `Set recommend=true ONLY when readable>=2 AND obvious_win>=2. Return ALL candidates, ranked best-first.\n\n` +
    `Candidates:\n${lines}`
}

phase('Judge')
log(`judging ${GROUPS.length} (stage,scene) groups`)
const judged = await parallel(GROUPS.map(g => () =>
  agent(judgePrompt(g), { label: `judge:${g.stage}/${g.scene}`, phase: 'Judge', schema: RANK_SCHEMA })
    .then(r => ({ g, ranked: (r && r.ranked) || [] }))))

// flatten recommended picks, attach dir/stage/scene
const recs = []
for (const j of judged.filter(Boolean)) {
  const idset = new Set(j.g.ids)
  for (const r of j.ranked) {
    if (!r.recommend || !idset.has(r.id)) continue
    recs.push({ ...r, stage: j.g.stage, scene: j.g.scene, path: `${j.g.dir}/${r.id}.png`,
                combined: (r.readable || 0) + (r.obvious_win || 0) })
  }
}
log(`${recs.length} recommended across ${new Set(recs.map(r => r.stage + '/' + r.scene)).size} scenes`)

// Verify: correctly-framed adversarial recheck of each recommended pick.
// The claim is NOT "Ours beats every baseline" (MH-3DGS is a strong competitor and may tie Ours by
// design) — it is "the WEAK baselines collapse under this biased view while Ours stays faithful".
phase('Verify')
const verified = await parallel(recs.map(r => () =>
  agent(`Verify this qualitative crop for a 3DGS paper figure. Read the image: ${r.path}\n` +
        `Layout: [full GT with red box] | [GT crop] | [Ours, green border] | [baseline crops]: ` +
        `Hier-GS, Scaffold-GS, MH-3DGS, MVGS` + (r.stage === 'discontinuous' ? ', and 3DGS.\n' : '.\n') +
        `IMPORTANT framing: the figure does NOT claim Ours beats every baseline. MH-3DGS is a strong ` +
        `competitor and may look as good as Ours — that is expected and FINE. The claim is: under this ` +
        `biased / held-out viewpoint the WEAK baselines collapse while Ours stays faithful to the GT crop.\n\n` +
        `Set confirm=true ONLY if ALL THREE hold:\n` +
        `  1) readable — recognizable structure (object edges, contours, architecture, furniture, patterns), ` +
        `NOT flat sky / flat wall / shadow / undifferentiated foliage mush;\n` +
        `  2) at least TWO of the WEAK baselines (Hier-GS, Scaffold-GS, MVGS` +
        (r.stage === 'discontinuous' ? ', 3DGS' : '') + `) show a clearly visible failure right on that ` +
        `structure — floater, smear, ghost/doubling, heavy blur, or missing geometry;\n` +
        `  3) Ours is itself FAITHFUL to the GT crop there — reasonably sharp and clean, NOT blurry, ` +
        `washed-out, hazy, or smeared.\n\n` +
        `REJECT if Ours is itself blurry/hazy/smeared, or if the weak baselines look basically fine. ` +
        `Ignore MH-3DGS when deciding. obvious_win = how striking the weak-baseline failure is (0-3). ` +
        `Give readable, obvious_win, a one-sentence caption (structure + how weak baselines fail + how Ours is faithful), and a reason.`,
        { label: `verify:${r.stage}/${r.scene}/${r.id}`, phase: 'Verify', schema: VERDICT_SCHEMA })
    .then(v => ({ ...r, verdict: v }))))

const confirmed = verified.filter(Boolean)
  .filter(r => r.verdict && r.verdict.confirm)
  .map(r => ({
    id: r.id, stage: r.stage, scene: r.scene, path: r.path,
    readable: r.verdict.readable ?? r.readable,
    obvious_win: r.verdict.obvious_win ?? r.obvious_win,
    artifact: r.verdict.artifact || r.artifact || '',
    caption: r.verdict.caption || r.caption || '',
    score: (r.verdict.readable ?? r.readable ?? 0) + (r.verdict.obvious_win ?? r.obvious_win ?? 0),
  }))
  .sort((a, b) => b.score - a.score)

log(`confirmed ${confirmed.length} crops after adversarial verify`)
return { confirmed }
