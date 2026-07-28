"""Generate the annotatable figure-review artifact (self-contained HTML).

Embeds each built figure as a base64 PNG, with a description (what it shows /
what it conveys) and an itemized key (terms, visual cues, axes), plus a per-
section note layer whose notes copy out as markdown to paste back to Claude.

    uv run --with pymupdf python scripts/build_review_artifact.py --out review.html
"""
import argparse
import base64
import pathlib

import fitz  # pymupdf

HERE = pathlib.Path(__file__).resolve().parent
FIGDIR = HERE.parent / "figures"

FIGS = [
    dict(
        file="burried", slide="Slide 2 - live example", tag="content",
        title="The redundancy problem, on real data",
        shows="Marginal enrichment (volcano: &minus;log&#8321;&#8320; p vs log&#8321;&#8320; odds ratio) over all gene "
              "sets, from the logistic-SuSiE GSEA analysis (2022 PCB talk). Red = rejected (significant); "
              "triangles = the sets that are truly \"active\" (driving the signal).",
        convey="Real <b>live example</b> of the redundancy headache: marginal testing flags <b>hundreds</b> "
               "of significant gene sets (red), because overlapping / correlated sets all fire for the same "
               "underlying signal. The few truly driving sets (triangles) are <b>buried</b> in that cloud.",
        key=[
            "<b>red</b> = marginally significant (rejected); <b>grey</b> = not significant; <b>triangles</b> = truly active sets.",
            "the sheer number of red points is the problem - you can't read the biology off a list of hundreds of redundant hits.",
            "historical image from the 2022 PCB deck (see resources/PROVENANCE.md); not regenerated here.",
        ],
    ),
    dict(
        file="tcellenrich", slide="Slide 3 - live example", tag="content",
        title="SuSiE resolves the redundancy",
        shows="T-cell GO:BP enrichment (logistic vs linear SuSiE): the blue cloud of redundant marginal "
              "enrichments, with the SuSiE credible-set components L1-L4 marked as large coloured points.",
        convey="Same data, SuSiE's answer: it assigns the redundant cloud to a <b>few credible sets</b> "
               "(L1-L4). Redundant signals are prioritized <b>within</b> one set; complementary signals land "
               "in <b>separate</b> sets. This is the payoff that motivates the whole method - and ties back to "
               "Logistic SuSiE (&sect;1) / GLM SuSiE (&sect;3).",
        key=[
            "<b>blue</b> = marginally enriched (the redundant cloud); <b>large coloured points</b> = SuSiE components L1-L4.",
            "logistic vs linear panels: the credible sets are stable across the two link choices.",
            "historical image from the 2022 PCB deck; real T-cell data.",
        ],
    ),
    dict(
        file="fig_two_regimes", slide="Motivation", tag="spotlight",
        title="Two jobs for enrichment - and where the 2&times;2 fails",
        shows="Two gene sets vs the genome-wide null. <b>A</b>: a strong result - a few large-effect "
              "genes in a small set (a bump past &tau;). <b>B</b>: a coordinated shift - all ~50 genes "
              "nudged right by +0.45, none crossing &tau;.",
        convey="Enrichment does two jobs: characterize a <b>strong result</b> (a few big effects) or "
               "detect a <b>small coordinated shift</b>. The threshold 2&times;2 / hit-list does the first "
               "(A: 7 hits vs 1.1 expected) but is <b>blind to the second</b> (B: 3 hits vs 1.1 - looks "
               "null) even though the set is really shifted. Thresholding discards exactly the "
               "aggregation signal - which is the motivation for modeling richer resolutions, and it "
               "sets up the richer 'cost of discretizing' example (slide 10).",
        key=[
            "<b>dashed grey</b> = genome-wide null; <b>crimson</b> = the gene set's z-score density; the dotted line is the threshold &tau;.",
            "<b>hits &gt; &tau;</b> vs the null expectation is all the 2&times;2 sees; <b>mean shift</b> is what a continuous / rank method sees.",
            "A is detected by both; B is missed by the 2&times;2 but caught by the shift - the coarser the resolution, the more of job B you lose.",
            "x-axis = gene-level z-score. Placed early in the deck, before the resolution ladder.",
        ],
    ),
    dict(
        file="fig06_resolution_spectrum", slide="Slide 6", tag="spotlight",
        title="One dataset, three resolutions",
        shows="One simulated two-group dataset (22 genes) ranked by the <b>Wald statistic z&sup2;</b>, "
              "shown at three resolutions in a row: (a) scores (z&sup2; magnitude + order), (b) ranks "
              "(order only), (c) binary (above/below a Wald threshold).",
        convey="Each step down the resolution ladder discards information; the binary view is the "
               "coarsest - a single cut.",
        key=[
            "ranked by <b>z&sup2; (Wald statistic)</b>, consistent with the &chi;&sup2; / inv-&chi;&sup2; misspecification analysis.",
            "<b>null (cool grey)</b> / <b>non-null (crimson)</b>: true class of each gene - now using <b>colour-menu option 3</b>, per your pick.",
            "<b>(a)</b> bar height = z&sup2;, dashed line = threshold. <b>(b)</b> a descending <b>staircase</b> = order only (magnitude discarded). <b>(c)</b> = 1(z&sup2; &gt; &tau;).",
            "x-axis (shared): genes ranked by z&sup2;, most significant on the left. Three square panels.",
        ],
    ),
    dict(
        file="fig08_misspecification", slide="Slide 8", tag="spotlight",
        title="Misspecifying f&#8321; is the real risk (not identifiability)",
        shows="Two panels. <b>Left</b>: the true scale-driven marginal (black) vs the <b>estimated "
              "location family</b> (crimson dashed) - it can't reach the scale tails (shaded misfit; the "
              "arrow points to the missed left tail). <b>Right</b>: mean causal PIP over 150 real SER sims "
              "for the oracle, the correctly-specified scale-family, the <b>both-estimated location-scale "
              "family</b>, the misspecified location-family, and cox-reversed (which models no f&#8321;).",
        convey="You were right: with a parametric f&#8321; the model <i>is</i> identifiable - the practical "
               "danger is assuming the <b>wrong family</b>. On scale-driven data the location-family fit "
               "<b>collapses to 0.04 PIP</b>, while the correct scale-family (0.76), the both-estimated "
               "loc-scale family (0.76), the oracle (0.79), and cox-reversed (0.75) all hold. Estimating "
               "both loc and scale recovers it; only the <i>restricted wrong</i> family fails. The rank "
               "method never models f&#8321;, so misspecification can't touch it.",
        key=[
            "<b>left</b>: solid = true marginal (scale-driven), dashed = estimated location family; shaded + arrow = the (left) tail mass the wrong family misses.",
            "<b>right</b> bars (repo method colours): oracle (rose), scale-fam / correct (tomato), loc-scale / both-estimated (vermillion), loc-fam / misspecified (crimson), cox-rev / no-f&#8321; (amber); &plusmn;1 SE.",
            "the restricted misspecified two-group loses essentially all power; the flexible family and the rank method do not.",
            "real pilot: fig08_experiment.py &rarr; fig08_data.json &rarr; this plot. scale_1.5, 150 sims.",
        ],
    ),
    dict(
        file="fig09_logistic_estimand", slide="Slide 9", tag="content",
        title="What logistic actually estimates",
        shows="The null density f&#8320; and non-null density f&#8321;, a binarization threshold &tau;, and the "
              "shaded false-positive and false-negative regions it creates.",
        convey="Binarizing changes the estimand: the logistic coefficient is the log-odds of "
               "<i>exceeding &tau;</i>, not of being differentially expressed. Overlap of f&#8320; and f&#8321; "
               "produces false positives and false negatives.",
        key=[
            "<b>f&#8320; (cool grey)</b> = null density N(0,1); <b>f&#8321; (crimson)</b> = non-null density.",
            "<b>&tau; (dashed)</b> = the binarization threshold.",
            "the shaded regions are the two error types - <b>false positives</b> (null above &tau;) and <b>false negatives</b> (non-null below &tau;); now identified in the <b>legend</b> rather than by in-plot arrows/labels.",
            "x-axis = z-score, y-axis = density.",
        ],
    ),
    dict(
        file="fig10_pip_vs_threshold", slide="Slide 10", tag="spotlight",
        title="The cost of discretizing",
        shows="Mean causal-variable PIP over ~200 <b>real SER simulations</b> (scale-driven signal, so "
              "f&#8321; is poorly identified), for the two-group oracle / estimated and cox-reversed "
              "(threshold-free horizontal lines) and for cox (censored) and logistic at thresholds "
              "&tau; = 1..4 (the curves).",
        convey="The threshold-free methods cluster high (~0.76-0.80, oracle on top). The binarizing "
               "methods depend on &tau;: both peak at &tau;=2 and collapse by &tau;=4 - too low dilutes, "
               "too high loses power. That gap between the flat cluster and the humped curves is the "
               "cost of discretizing.",
        key=[
            "<b>mean causal PIP</b> = posterior inclusion probability of the true causal gene set, averaged over ~200 simulated datasets (real two-group / cox / logistic SER fits).",
            "<b>value labels 0.80 / 0.77 / 0.76</b> mark the three threshold-free methods - oracle (rose), estimated (vermillion dashed), cox-reversed (amber dotted) - nearly tied here, oracle highest.",
            "<b>green squares</b> = cox (regular, censored) at threshold &tau;; <b>blue circles</b> = logistic at &tau;; error bars are &plusmn;1 SE. Legend and settings box sit <b>outside</b> the plot frame.",
            "x-axis = threshold &tau;, y-axis = mean causal PIP.",
            "<b>Regime note:</b> the two hard orderings hold here (oracle &gt; estimated, oracle &gt; cox-reversed). cox-reversed only <i>beats</i> estimated at lower SNR - where the oracle also drops - so the full oracle &gt; cox-reversed &gt; estimated chain doesn't co-occur in this design (see the fig10 pilot note).",
        ],
    ),
    dict(
        file="fig14_hazard_loc_scale", slide="Slide 14", tag="spotlight",
        title="The PH assumption breaks (location vs scale)",
        shows="Two panels - a <b>pure-location</b> non-null (f&#8321; = N(3, 0.01&sup2;), a near point mass) and "
              "a <b>pure-scale</b> non-null (f&#8321; = N(0, 2&sup2;)). Each plots the log hazard ratio versus "
              "arrival time for the forward ordering (cox, green) and the reverse ordering (cox-reversed, "
              "amber).",
        convey="Neither ordering satisfies proportional hazards (that would be a flat line). But the "
               "forward ordering's violation blows up for the most informative genes, while the reverse "
               "ordering's is milder and bounded - so reverse is the safer fit.",
        key=[
            "parameterized by <b>f&#8321; = N(loc, scale&sup2;)</b> on the true effect; the observed non-null z is N(loc, 1+scale&sup2;), so the hazard uses &sigma; = &radic;(1+scale&sup2;). (So &sigma; is set by the f&#8321; scale, not read off the marginal.)",
            "<b>forward (cox, green)</b>: arrival time T = 1/z&sup2; (most significant arrive earliest) &rarr; InvChi&sup2;-type.",
            "<b>reverse (cox-reversed, amber)</b>: arrival time S = z&sup2; (most significant arrive latest) &rarr; Chi&sup2;-type; plotted as &minus;log R_S so both curves sit above zero.",
            "the <b>green</b> curve exploding as t&rarr;0 is the violation landing on the <b>most significant</b> genes; a flat line at 0 (dotted) would be exact PH.",
        ],
    ),
    dict(
        file="fig15_late_arrival", slide="Slide 15", tag="spotlight",
        title="Late arrivals dominate the score",
        shows="The <b>total</b> perturbation of the cox score, "
              "&Delta;g(k) = 1 &minus; (H_n &minus; H_{n&minus;k}), from swapping one observation at arrival "
              "position k into the enriched group (n = 1000).",
        convey="An early swap barely moves the score (&Delta;g = 1 &minus; 1/n &asymp; +1, O(1)); the last "
               "arrival swings it by &asymp; &minus;log n (O(log n)). Late arrivals dominate.",
        key=[
            "<b>&Delta;g(k)</b> = count term (+1, the same for any k) &minus; position term (H_n &minus; H_{n&minus;k}). H_m = &Sigma;_{i=1}^m 1/i is the m-th harmonic number.",
            "<b>early (k=1)</b>: 1 &minus; 1/n &asymp; +1.  <b>late (k=n)</b>: 1 &minus; H_n &asymp; &minus;log n.  (This corrects the earlier version, which showed only the position term and read 1/n at k=1.)",
            "k is <b>1-indexed</b>; k=1 is the first (earliest) arrival. The plot is agnostic to how the data were ordered - it shows influence vs arrival position.",
            "x-axis = arrival position k; y-axis = score perturbation &Delta;g(k).",
        ],
    ),
    dict(
        file="figB2_cox_poisson", slide="Backup B2", tag="deriv",
        title="Exponential arrivals: where enriched genes land",
        shows="The enriched gene's background quantile U = F&#8320;(Y) ~ Beta(1, &rho;), "
              "&rho; = &lambda;&#8321;/&lambda;&#8320;, for enriched-faster (&rho;=4) and enriched-slower "
              "(&rho;=0.25). Single PDF panel; the enriched-slower <b>left tail [0, 0.2]</b> and the "
              "enriched-faster <b>right tail [0.8, 1.0]</b> are shaded, with their tail probabilities.",
        convey="The point (now framed the way you asked): the enriched-slower <b>early tail is much "
               "heavier</b> than the enriched-faster <b>late tail</b> - P(U&lt;0.2)=0.054 for slower vs "
               "P(U&gt;0.8)=0.0016 for faster. A slow (reverse-favouring) gene can plausibly arrive early; "
               "a fast (forward-favouring) gene almost never arrives late.",
        key=[
            "<b>&rho; = &lambda;&#8321;/&lambda;&#8320;</b> rate ratio; <b>U ~ Beta(1, &rho;)</b>; <b>P(Y&lt;X) = &rho;/(1+&rho;)</b>.",
            "<b>colours = ordering favoured</b>: forward (cox <span style='color:#009E73'>green</span>), reverse (cox-reversed <span style='color:#E69F00'>amber</span>).",
            "shaded areas are the two tails being compared; y capped so the shapes are comparable. 4:3 landscape, tail probabilities placed above their curves.",
        ],
    ),
    dict(
        file="fig08_f1_identifiability", slide="Slide 8 - backup", tag="deriv",
        title="Set aside: the identifiability ridge",
        shows="The earlier framing (scale parameterization): decompositions of two scenarios (left) and "
              "the marginal log-likelihood ridge over (&pi;, &theta;) (right).",
        convey="Kept for reference only. As you noted, with a parametric f&#8321; this is <b>finite-sample "
               "weak identification</b>, not true non-identifiability - the ridge shrinks as n grows. The "
               "misspecification framing above is the real story; this stays as a backup. (The deeper "
               "genuine non-identifiability - the null proportion &pi;&#8320; with a <i>flexible</i> f&#8321; - is a "
               "separate result I can draw if you want it.)",
        key=[
            "loc / moment parameterization variants are still buildable on demand (see the fig08_variant rule), just no longer front-and-centre.",
        ],
    ),
]

TAG_LABEL = {"spotlight": "Spotlight", "content": "Content", "deriv": "Derivation",
             "option": "Option"}


RESDIR = HERE.parent / "resources"


def img_data_uri(name, dpi=140):
    # generated figures live in figures/*.pdf; historical example images in resources/*.png
    pdf = FIGDIR / (name + ".pdf")
    if pdf.exists():
        png = fitz.open(pdf)[0].get_pixmap(dpi=dpi).tobytes("png")
    else:
        png = (RESDIR / (name + ".png")).read_bytes()
    return "data:image/png;base64," + base64.b64encode(png).decode()


STYLE = """
:root{
  --ink:#16202E; --ink-soft:#33414F; --paper:#F4F6F8; --panel:#FFFFFF; --card:#FFFFFF;
  --hair:#DDE3E9; --muted:#66788A; --accent:#0B6E75; --accent-ink:#075158;
  --amber:#CC7A34; --danger:#B0552C;
  --spot:#2E8B62; --spot-bg:#E6F1EB; --content:#3D6E8C; --content-bg:#E6EEF3;
  --deriv:#5C6BA8; --deriv-bg:#E7E9F3;
  --serif:Charter,"Bitstream Charter","Iowan Old Style","Palatino Linotype",Georgia,ui-serif,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --mono:"SF Mono",ui-monospace,"JetBrains Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.55;-webkit-font-smoothing:antialiased;}
.wrap{max-width:940px;margin:0 auto;padding:0 24px}
.masthead{padding:56px 0 26px;border-bottom:2px solid var(--ink);margin-bottom:8px}
.eyebrow{font-family:var(--mono);font-size:12px;letter-spacing:.15em;text-transform:uppercase;
  color:var(--accent-ink);display:flex;gap:12px;align-items:center}
.eyebrow::after{content:"";height:1px;width:46px;background:var(--amber)}
.masthead h1{font-family:var(--serif);font-weight:700;font-size:clamp(30px,4.5vw,46px);
  letter-spacing:-.01em;margin:16px 0 0;line-height:1.05;text-wrap:balance}
.masthead p{font-size:17px;color:var(--ink-soft);margin:16px 0 0;max-width:66ch}
.masthead .meta{margin-top:18px;font-family:var(--mono);font-size:12px;color:var(--muted);
  display:flex;gap:22px;flex-wrap:wrap}
.masthead .meta b{color:var(--ink)}

figure.fig{margin:0}
.figcard{background:var(--panel);border:1px solid var(--hair);border-radius:14px;
  padding:22px 24px 20px;margin:30px 0;box-shadow:0 16px 34px -28px #16202E66}
.figcard .top{display:flex;align-items:baseline;gap:12px;margin-bottom:4px}
.figcard .slide{font-family:var(--mono);font-size:11px;font-weight:600;color:var(--muted);
  border:1px solid var(--hair);border-radius:6px;padding:2px 8px}
.tag{font-family:var(--mono);font-size:10.5px;font-weight:700;letter-spacing:.04em;
  padding:2px 8px;border-radius:5px;text-transform:uppercase}
.tag.spotlight{color:var(--spot);background:var(--spot-bg)}
.tag.content{color:var(--content);background:var(--content-bg)}
.tag.deriv{color:var(--deriv);background:var(--deriv-bg)}
.tag.option{color:#8A5A00;background:#F6ECD9}
.figcard h2{font-family:var(--serif);font-size:23px;font-weight:600;letter-spacing:-.01em;
  margin:6px 0 2px;text-wrap:balance}
.figimg{margin:14px 0 4px;text-align:center;background:#fff;border:1px solid var(--hair);
  border-radius:10px;padding:10px;overflow-x:auto}
.figimg img{max-width:100%;height:auto;display:block;margin:0 auto}
.blocks{display:grid;grid-template-columns:1fr 1fr;gap:20px 28px;margin-top:16px}
@media (max-width:680px){.blocks{grid-template-columns:1fr}}
.block h3{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:var(--accent-ink);margin:0 0 6px}
.block p{margin:0;font-size:14.5px;color:var(--ink-soft)}
.block.wide{grid-column:1 / -1}
.keylist{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:7px}
.keylist li{font-size:13.5px;color:var(--ink-soft);padding-left:16px;position:relative;line-height:1.45}
.keylist li::before{content:"";position:absolute;left:0;top:8px;width:6px;height:6px;
  border-radius:50%;background:var(--amber)}
.keylist b{color:var(--ink)}
.foot{margin:44px 0 10px;padding:22px 0 90px;border-top:1px solid var(--hair);
  color:var(--muted);font-size:13.5px}
.foot b{color:var(--ink)}
"""


def build_section(fig):
    tag = fig["tag"]
    key_items = "\n".join(f"      <li>{k}</li>" for k in fig["key"])
    anchor = fig["file"]
    return f"""
  <figure class="fig">
    <div class="figcard">
      <div class="top">
        <span class="slide">{fig['slide']}</span>
        <span class="tag {tag}">{TAG_LABEL[tag]}</span>
      </div>
      <h2>{fig['title']}</h2>
      <div class="figimg"><img alt="{fig['title']}" src="{img_data_uri(fig['file'])}"></div>
      <div class="blocks">
        <div class="block"><h3>What it shows</h3><p>{fig['shows']}</p></div>
        <div class="block"><h3>What it conveys</h3><p>{fig['convey']}</p></div>
        <div class="block wide"><h3>Key - terms, cues, axes</h3>
          <ul class="keylist">
{key_items}
          </ul>
        </div>
      </div>
    </div>
    <div class="notes" data-anchor="{anchor}" data-title="{fig['slide']} - {fig['title']}"></div>
  </figure>"""


ANNO_STYLE = """
<style>
  .anno-bar{position:fixed;top:14px;right:14px;z-index:50;display:flex;align-items:center;gap:8px;
    background:var(--panel,#fff);border:1px solid var(--hair,#e2e6ec);border-radius:999px;padding:6px 8px 6px 15px;
    box-shadow:0 6px 20px rgba(27,36,48,.14);font-family:var(--sans);font-size:12.5px;}
  .anno-bar .cnt{color:var(--muted,#616b7a);} .anno-bar .cnt b{color:var(--ink,#1b2430);font-variant-numeric:tabular-nums;}
  .anno-bar button{font-family:inherit;font-size:12.5px;border:1px solid var(--hair,#e2e6ec);background:var(--paper,#f5f6f8);
    color:var(--ink,#1b2430);border-radius:999px;padding:5px 13px;cursor:pointer;}
  .anno-bar button:hover{border-color:var(--accent,#0B6E75);color:var(--accent,#0B6E75);}
  .anno-bar button.send{background:var(--accent,#0B6E75);color:#fff;border-color:var(--accent,#0B6E75);}
  .anno-bar button.send:hover{color:#fff;filter:brightness(1.08);}
  .anno-bar button.warn{border:none;color:var(--muted,#616b7a);padding:5px 8px;}
  .anno-bar button.warn:hover{color:var(--danger,#a9502b);}
  .anno-intro{font-family:var(--sans);font-size:13.5px;color:var(--muted,#616b7a);
    background:var(--panel,#fff);border:1px solid var(--hair,#e2e6ec);border-left:3px solid var(--accent,#0B6E75);
    border-radius:10px;padding:13px 17px;margin:18px 0 0;}
  .anno-intro b{color:var(--ink,#1b2430);}
  .notes{margin:14px 0 2px;display:grid;gap:9px;}
  .notes-head{display:flex;align-items:center;gap:14px;}
  .notes-lbl{font-family:var(--sans);font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted,#616b7a);}
  .note-add{font-family:var(--sans);font-size:12.5px;border:1px dashed var(--hair,#e2e6ec);background:transparent;
    color:var(--accent,#0B6E75);border-radius:8px;padding:5px 13px;cursor:pointer;}
  .note-add:hover{border-color:var(--accent,#0B6E75);border-style:solid;}
  .note-card{position:relative;background:#fffcf3;border:1px solid #ece3c9;border-radius:8px;
    padding:11px 36px 11px 15px;font-family:var(--sans);font-size:14px;color:var(--ink,#1b2430);white-space:pre-wrap;}
  .note-card .note-del{position:absolute;top:5px;right:8px;border:none;background:transparent;color:var(--muted,#616b7a);font-size:18px;line-height:1;cursor:pointer;padding:2px 5px;}
  .note-card .note-del:hover{color:var(--danger,#a9502b);}
  .note-edit{display:grid;gap:8px;}
  .note-input{font-family:var(--sans);font-size:14px;width:100%;border:1px solid var(--hair,#e2e6ec);border-radius:8px;padding:10px 12px;resize:vertical;background:var(--panel,#fff);color:var(--ink,#1b2430);}
  .note-input:focus{outline:2px solid var(--accent,#0B6E75);outline-offset:1px;border-color:var(--accent,#0B6E75);}
  .note-bar{display:flex;gap:8px;}
  .note-bar button{font-family:var(--sans);font-size:13px;border-radius:8px;padding:6px 15px;cursor:pointer;border:1px solid var(--hair,#e2e6ec);}
  .note-save{background:var(--accent,#0B6E75);color:#fff;border-color:var(--accent,#0B6E75);}
  .note-cancel{background:var(--paper,#f5f6f8);color:var(--muted,#616b7a);}
  .modal-bg{position:fixed;inset:0;background:rgba(27,36,48,.44);z-index:60;display:flex;align-items:center;justify-content:center;padding:24px;}
  .modal{background:var(--panel,#fff);border-radius:14px;max-width:640px;width:100%;padding:24px;box-shadow:0 24px 70px rgba(27,36,48,.32);}
  .modal h3{font-family:var(--sans);margin:0 0 4px;font-size:18px;color:var(--ink,#1b2430);}
  .modal p{font-family:var(--sans);font-size:13.5px;color:var(--muted,#616b7a);margin:0 0 13px;}
  .modal textarea{width:100%;height:240px;font-family:var(--mono);font-size:12.5px;border:1px solid var(--hair,#e2e6ec);border-radius:8px;padding:12px;resize:vertical;color:var(--ink,#1b2430);background:var(--paper,#f5f6f8);}
  .modal .mbar{display:flex;gap:8px;justify-content:flex-end;margin-top:13px;}
  .modal .mbar button{font-family:var(--sans);font-size:13px;border-radius:8px;padding:7px 17px;cursor:pointer;border:1px solid var(--hair,#e2e6ec);background:var(--paper,#f5f6f8);color:var(--ink,#1b2430);}
  .modal .mbar button.primary{background:var(--accent,#0B6E75);color:#fff;border-color:var(--accent,#0B6E75);}
  @media print{.anno-bar,.note-add,.note-del{display:none;}}
</style>
"""

# Script adapted from artifact-notes snippet; Export relabeled "Copy for Claude".
ANNO_SCRIPT = r"""
<script>
(function(){
  var KEY='gsea_figs_review_v1', store={}, persist=true;
  try{ var raw=localStorage.getItem(KEY); store=raw?JSON.parse(raw):{}; localStorage.setItem(KEY, raw||'{}'); }
  catch(e){ persist=false; }
  function total(){ var n=0; for(var k in store) n+=(store[k]||[]).length; return n; }
  var barCnt;
  function updateBar(){ if(barCnt) barCnt.innerHTML='<b>'+total()+'</b> note'+(total()===1?'':'s'); }
  function save(){ if(persist){ try{ localStorage.setItem(KEY, JSON.stringify(store)); }catch(e){ persist=false; } } updateBar(); }
  function el(t,c,txt){ var e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e; }
  function render(c){
    var a=c.dataset.anchor, list=store[a]||[]; c.innerHTML='';
    var head=el('div','notes-head'); head.appendChild(el('span','notes-lbl','Notes on '+c.dataset.title));
    var add=el('button','note-add','+ Add note'); add.onclick=function(){ edit(c); };
    head.appendChild(add); c.appendChild(head);
    list.forEach(function(n,i){ var card=el('div','note-card',n.text);
      var d=el('button','note-del','×'); d.title='Delete note';
      d.onclick=function(){ store[a].splice(i,1); if(!store[a].length) delete store[a]; save(); render(c); };
      card.appendChild(d); c.appendChild(card); });
  }
  function edit(c){ var a=c.dataset.anchor; var w=el('div','note-edit');
    var ta=el('textarea','note-input'); ta.rows=2; ta.placeholder='Note on '+c.dataset.title+'… (⌘/Ctrl+Enter to save)';
    var bar=el('div','note-bar'); var s=el('button','note-save','Save'), x=el('button','note-cancel','Cancel');
    bar.appendChild(s); bar.appendChild(x); w.appendChild(ta); w.appendChild(bar); c.appendChild(w); ta.focus();
    s.onclick=function(){ var t=ta.value.trim(); if(t){ (store[a]=store[a]||[]).push({text:t,ts:0}); save(); } render(c); };
    x.onclick=function(){ render(c); };
    ta.addEventListener('keydown',function(ev){ if((ev.metaKey||ev.ctrlKey)&&ev.key==='Enter'){ s.onclick(); } });
  }
  var anchors=[].slice.call(document.querySelectorAll('.notes'));
  anchors.forEach(render);
  var bar=el('div','anno-bar'); barCnt=el('span','cnt'); bar.appendChild(barCnt);
  var exp=el('button','send','Copy for Claude'); exp.onclick=exportNotes; bar.appendChild(exp);
  var clr=el('button','warn','Clear'); clr.onclick=function(){ if(confirm('Delete all notes on this page?')){ store={}; save(); anchors.forEach(render); } };
  bar.appendChild(clr); document.body.appendChild(bar); updateBar();
  function exportNotes(){
    var md='# GSEA figure review — notes\n\n', any=false;
    anchors.forEach(function(c){ var list=store[c.dataset.anchor]||[]; if(list.length){ any=true;
      md+='## '+c.dataset.title+'\n'; list.forEach(function(n){ md+='- '+n.text.replace(/\n/g,' ')+'\n'; }); md+='\n'; } });
    if(!any) md+='_(no notes yet — click “+ Add note” under a figure)_\n';
    var bg=el('div','modal-bg'), m=el('div','modal');
    m.appendChild(el('h3',null,'Send these notes to Claude'));
    m.appendChild(el('p',null,'Hosted artifacts can’t message the chat directly, so this is the hand-off: hit Copy, then paste into the chat. I’ll fold each note into the figure and redeploy.'));
    var ta=el('textarea'); ta.value=md; m.appendChild(ta);
    var mb=el('div','mbar');
    var cp=el('button','primary','Copy'); cp.onclick=function(){ ta.select(); try{ navigator.clipboard.writeText(md); }catch(e){ try{ document.execCommand('copy'); }catch(_){} } cp.textContent='Copied — now paste into chat'; };
    var cl=el('button',null,'Close'); cl.onclick=function(){ document.body.removeChild(bg); };
    mb.appendChild(cp); mb.appendChild(cl); m.appendChild(mb); bg.appendChild(m);
    bg.onclick=function(ev){ if(ev.target===bg) document.body.removeChild(bg); };
    document.body.appendChild(bg); ta.focus();
  }
  if(!persist){ var intro=document.querySelector('.anno-intro');
    if(intro) intro.innerHTML='<b>Heads up:</b> this browser is blocking local storage in the artifact frame, so notes last only until you reload. Add notes, then hit <b>Copy for Claude</b> before leaving.'; }
})();
</script>
"""


def main(out: str) -> None:
    sections = "\n".join(build_section(f) for f in FIGS)
    intro = ('<div class="anno-intro"><b>Leave comments here.</b> Click <b>+ Add note</b> under any '
             'figure to attach a comment - they save in this browser. When you’re done, hit '
             '<b>Copy for Claude</b> (top-right) and paste into the chat; I’ll fold each note into '
             'the matching figure and redeploy.</div>')
    html = f"""<title>GSEA figures - review</title>
<style>{STYLE}</style>
{ANNO_STYLE}
<div class="wrap">
  <div class="masthead">
    <div class="eyebrow">Committee meeting &middot; Revisiting GSEA &middot; local figures</div>
    <h1>Figure review - the eight local figures</h1>
    <p>All of Track A, built and provenance-tracked (each has a <code>.prov.json</code> sidecar).
    These are the small, exact figures that carry the argument without waiting on the migration.
    Read the description + key under each, and leave comments.</p>
    <div class="meta"><span><b>8</b> figures</span><span><b>6</b> spotlight &middot; 2 derivation</span>
      <span>all <b>local</b>, reproducible</span></div>
  </div>
  {intro}
{sections}
  <div class="foot"><b>Note:</b> figures are real computations (fixed seeds), not the migrated
  pipeline. fig10 (cost of discretizing) is a score-test proxy - trust its shape, not exact heights.
  Colours: <b>teal</b> = null / two-group / forward, <b>orange</b> = non-null / logistic / reverse.</div>
</div>
{ANNO_SCRIPT}
"""
    pathlib.Path(out).write_text(html)
    print("wrote", out, f"({len(html)//1024} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    main(ap.parse_args().out)
