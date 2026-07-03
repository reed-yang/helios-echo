#!/usr/bin/env python3
"""Build the event-switch eval CSV for the cfr / captions_30b_32f run, in the FULL 6-tag training format.

First 5 videos of example/prompt_interactive_helios_wander.csv (ids 1000/1001/1002/1003/1010, 6 events each),
rewritten into our training prompt format with ALL six tags AND matched to the training corpus's per-tag
CONTENT STYLE + LENGTH (measured on 20k train prompts):
  header ~4w (fixed) | event ~22w (15-32) | role ~27w (20-36) | Background ~23w (15-33) | style 1w | scene ~39w (29-51)
Content format mirrors training:
  <event>  : one descriptive sentence, third-person, present tense, person + action (no <ID_1> tag).
  <role>   : "One person. <ID_1> a <age/ethnicity>, <skin>, <hair/eyes>, wearing <clothing>." (const per video)
  <Background>: "<Indoors/Outdoors>, <setting>; <lighting>." (changes per segment)
  <style>  : "realistic" (const)
  <scene>  : "<Shot>, <angle>, <depth of field>. The camera <movement>. <ID_1> <detailed action>." (~39w)
<role> + <style> are held CONSTANT across a video's 6 events; <event>, <Background>, <scene> change per
segment as the character moves through the scene -> the model hard-switches at chunk boundaries.
Newlines joined to single lines (CSV-friendly). Output -> eval_prompts_cfr/evtsw.csv  (id,prompt_index,prompt).
1-min videos: 6 events x 7 chunks x 33 = 1386 frames at interpolate_time=7.
"""
import csv, os

HEADER = "high definition, normal speed"
STYLE = "realistic"

# id -> (role[const], [(event, background, scene) x6])
DATA = {
    "1000": (
        "One person. <ID_1> an elderly man in his 70s, light wrinkled skin, sparse white hair and a short white beard, wearing a worn linen jacket and a faded gray flat cap.",
        [
            ("An elderly man shuffles slowly into a cobblestone market square at first light, his shoulders hunched as he glances around at the waking stalls.",
             "Outdoors, a cobblestone market square at dawn, soft golden light spilling between old stone buildings; wooden market stalls and a few early shoppers sit blurred in the background.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera moves slowly backward, leading him as he advances. <ID_1> walks forward into the square with small unsteady steps, his head turning from side to side as he takes in the quiet stalls."),
            ("The elderly man strolls unhurriedly past a stall piled with oranges and ruby tomatoes, slowing and leaning in to inspect the bright produce.",
             "Outdoors, a produce stall stacked with glowing oranges and red tomatoes, warm morning light; out-of-focus shoppers and crates fill the background behind him.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks alongside him at a gentle pace. <ID_1> strolls past the stall, slowing his steps and turning his head to study the piled fruit, one hand half-raised toward it."),
            ("The elderly man stops before a bakery stall where loaves of country bread cool on a flour-dusted board, studying them with quiet interest.",
             "Outdoors, a rustic bakery stall, golden loaves of country bread on a flour-dusted wooden board; soft warm light and a blurred awning frame the background.",
             "Medium close-up, eye-level, shallow depth of field. The camera holds nearly steady, drifting in slightly. <ID_1> comes to a stop and leans forward, his eyes moving over the bread as he reaches out a slow, hesitant hand toward the nearest loaf."),
            ("The elderly man drifts past a street violinist tuning up beneath a striped awning, slowing almost to a halt as he tilts his head to listen.",
             "Outdoors, a shaded corner beneath a red-and-white striped awning, a blurred street violinist to one side; dappled morning light falls across the worn stones.",
             "Medium shot, eye-level, shallow depth of field. The camera follows him in a slow lateral drift. <ID_1> slows almost to a stop, tilting his head toward the music, a faint smile forming as his gaze settles on the unseen player."),
            ("The elderly man turns into a narrow alley strung with green vines and hanging laundry, moving deeper into the cool shadow between the walls.",
             "Outdoors, a narrow stone alley, climbing green vines and laundry lines overhead; cool shadow broken by scattered patches of bright light on the pavement.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera trails a few steps behind him down the alley. <ID_1> turns and walks away from camera into the passage, one hand brushing the stone wall as he steps over the uneven ground."),
            ("The elderly man emerges from the alley into a sun-flooded plaza ringed with stone benches, pausing as the bright light washes over his face.",
             "Outdoors, an open sunlit plaza ringed by stone benches and pale building facades; bright clear morning light floods the wide paved space around him.",
             "Wide shot, eye-level, shallow depth of field. The camera holds steady as he advances toward it. <ID_1> emerges from the shadow into bright sunlight, slowing as he steps into the open plaza, lifting his face slightly toward the warm glare."),
        ],
    ),
    "1001": (
        "One person. <ID_1> a young street photographer in his 20s, light skin, dark eyes, wearing a cherry-red knit beanie over an oversized army-green jacket, a camera slung across his chest.",
        [
            ("A young street photographer steps off a sidewalk into the neon glow of a Shibuya backstreet at dusk, lifting his camera as the lights flicker on.",
             "Outdoors, a narrow Tokyo backstreet at dusk, glowing neon signs and wet reflective pavement; a blurred evening crowd and shopfront lights fill the background.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera glides backward ahead of him through the neon street. <ID_1> steps off the curb into the colored light, camera raised to his chest, his eyes scanning the signs overhead as he walks."),
            ("The young photographer walks slowly past a pachinko parlor spilling amber light and chiming noise, glancing toward the bright glass doorway.",
             "Outdoors, the bright entrance of a pachinko parlor spilling warm amber light onto the pavement; blurred patrons and glowing signage crowd the background.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks beside him at a steady walk. <ID_1> moves slowly past the glowing doorway, turning his head toward the warm light, one hand resting on the camera at his chest as he peers inside."),
            ("The young photographer reaches the famous pedestrian scramble and steps off the curb as the signal turns green, raising his camera to the crowd.",
             "Outdoors, a wide neon-lit pedestrian crossing at night, towering bright screens and signage overhead; out-of-focus crowds stream across the painted lines.",
             "Wide shot, eye-level, shallow depth of field. The camera holds low and steady as the crowd flows around him. <ID_1> steps off the curb and begins crossing, lifting his camera toward the swirling pedestrians, turning slowly to follow the movement around him."),
            ("The young photographer ducks into a slim ramen alley lit by paper lanterns strung overhead, slowing to breathe in the drifting steam.",
             "Outdoors, a cramped ramen alley, red paper lanterns strung overhead and warm steam rising; blurred storefronts and glowing menus line the narrow passage.",
             "Medium shot, eye-level, shallow depth of field. The camera follows close behind as he enters the alley. <ID_1> ducks beneath the low lanterns and walks slowly inward, glancing up at the glowing paper shades, his camera lowered loosely in one hand."),
            ("The young photographer emerges past a row of crane-game arcades packed with glowing plush toys, turning his head to take in the bright cases.",
             "Outdoors, an arcade facade of glass crane-game cabinets packed with colorful glowing plush toys; bright saturated neon reflects off the polished glass.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks smoothly alongside him past the cabinets. <ID_1> walks past the glowing glass cases, turning his head to study the plush toys, slowing slightly as the colored light moves across his face."),
            ("The young photographer climbs the metal stairs onto an overpass crossing the main avenue, stopping at the rail as the city glows beneath him.",
             "Outdoors, a metal pedestrian overpass above a busy neon avenue at night; distant traffic light-trails and glowing towers stretch out beyond the railing.",
             "Wide shot, low angle, shallow depth of field. The camera tilts up to follow him as he climbs. <ID_1> ascends the metal stairs and steps onto the overpass, moving to the railing and leaning forward to look down at the glowing traffic below."),
        ],
    ),
    "1002": (
        "One person. <ID_1> a businesswoman in her 30s, light skin, dark hair pulled into a low bun, wearing a tan trench coat over pressed navy slacks, a slate-gray leather portfolio under one arm.",
        [
            ("A businesswoman strides into a glass-canyon financial district at midday, her portfolio tucked under one arm as she moves briskly through the crowd.",
             "Outdoors, a canyon of glass office towers at midday, bright overcast light reflecting off the facades; blurred pedestrians and slow traffic fill the background.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera glides backward ahead of her brisk pace. <ID_1> strides confidently forward through the district, her portfolio pressed to her side, her gaze fixed ahead as people pass blurred around her."),
            ("The businesswoman slips between two parked delivery vans and rejoins the sidewalk near a corner newsstand, glancing back at the traffic.",
             "Outdoors, a curbside gap between two parked delivery vans, a blurred corner newsstand beyond; flat midday daylight falls across the gray pavement.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks beside her through the gap. <ID_1> slips between the parked vans and steps back onto the sidewalk, glancing over her shoulder at the passing traffic before facing forward again."),
            ("The businesswoman rounds a corner past a tall revolving door at a glass-fronted bank, glancing at the slowly spinning panels as she passes.",
             "Outdoors, a glass bank facade with a slowly spinning revolving door, the street reflected in the panels; cool even daylight and blurred passersby behind.",
             "Medium shot, eye-level, shallow depth of field. The camera arcs slightly to follow her around the corner. <ID_1> rounds the corner at a steady pace, turning her head toward the revolving door, her portfolio swinging slightly as she continues past it."),
            ("The businesswoman crosses a small plaza where a low circular fountain pulses water into the air, slowing to skirt its splashing edge.",
             "Outdoors, a small open plaza with a low circular fountain, sparkling water jets catching the light; bright daylight and pale stone paving stretch around her.",
             "Wide shot, eye-level, shallow depth of field. The camera holds steady as she crosses the open space. <ID_1> walks across the plaza and skirts the edge of the fountain, glancing down at the rising water, slowing her stride to avoid the drifting spray."),
            ("The businesswoman reaches a subway entrance set in a granite tower and hesitates at the top of the stairs, glancing down into the descent.",
             "Outdoors, the granite base of an office tower with a subway entrance and descending stairs; signage and shaded cool light frame the shadowed opening.",
             "Medium shot, eye-level, shallow depth of field. The camera holds near her as she stops at the stairhead. <ID_1> reaches the top of the subway stairs and pauses, glancing down into the entrance, her hand tightening on the portfolio as she hesitates."),
            ("The businesswoman emerges onto a wider boulevard lined with leafy ginkgo trees, her pace easing as dappled light falls across her face.",
             "Outdoors, a wide boulevard lined with leafy green ginkgo trees, dappled sunlight on the pavement; blurred storefronts and slow traffic fill the background.",
             "Wide shot, eye-level, shallow depth of field. The camera glides backward as she steps into the open. <ID_1> emerges onto the tree-lined boulevard and slows her pace, her shoulders easing as the dappled light moves across her face and coat."),
        ],
    ),
    "1003": (
        "One person. <ID_1> a weathered street vendor in his 50s, sun-darkened skin, gray-flecked beard, wearing a sun-bleached blue djellaba and a faded red fez, gripping the handles of a hand-built wooden cart.",
        [
            ("A weathered street vendor pushes a wooden cart through the arched gate of a Marrakech medina at midmorning, leaning into the worn handles.",
             "Outdoors, an ornate arched medina gate, warm earthen walls glowing in dusty midmorning light; blurred passersby and market stalls crowd the background.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera glides backward through the gate ahead of him. <ID_1> pushes his wooden cart forward through the archway, leaning his weight into the handles, his eyes lifting to the carved stonework above as he passes."),
            ("The vendor maneuvers his cart past a spice stall stacked with cones of saffron, paprika, and cumin, steering carefully around the colorful mounds.",
             "Outdoors, a vivid spice stall with conical mounds of saffron, paprika and cumin in red and gold; warm light and a blurred shopkeeper fill the background.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks alongside the moving cart. <ID_1> steers the cart carefully past the spice cones, glancing down at the bright mounds, easing the wheels around them so the heaped powders are not disturbed."),
            ("The vendor ducks under a low blue-tiled archway, lifting the front rail of the cart to clear the narrow opening as he passes through.",
             "Outdoors, a low blue-tiled archway in a narrow medina lane, intricate patterns on the tiles; cool shadow broken by warm patches of light on the ground.",
             "Medium shot, eye-level, shallow depth of field. The camera follows close behind through the arch. <ID_1> ducks his head beneath the tiled archway and lifts the front rail of the cart, easing it over the threshold before lowering it and pushing onward into the lane."),
            ("The vendor slows at a small tiled fountain where children fill a clay jug, halting the cart and waiting patiently for them to finish.",
             "Outdoors, a small ornate tiled fountain set in a shaded corner, water and a clay jug catching the light; blurred children and soft warm light behind.",
             "Medium shot, eye-level, shallow depth of field. The camera holds steady on him beside the cart. <ID_1> brings the cart to a halt and waits, one hand resting on the handle, watching the fountain patiently as the blurred children fill their jug at the water."),
            ("The vendor pushes the cart through a textile section hung with bolts of indigo, crimson, and emerald wool, weaving between the draped fabrics.",
             "Outdoors, a textile lane hung with bolts of indigo, crimson and emerald wool, rich saturated color; warm light filters through the draped fabric overhead.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks with the cart down the draped lane. <ID_1> pushes the cart between the hanging fabrics, turning the handles to weave through the narrow gaps, the colored wool brushing past his shoulders as he goes."),
            ("The vendor wheels his cart into the open main square where snake charmers and crowds gather, slowing as the bustle opens up before him.",
             "Outdoors, a wide open medina square at midday, distant crowds and market activity spread across it; bright dusty light fills the broad sunlit space.",
             "Wide shot, eye-level, shallow depth of field. The camera holds steady as he rolls into view. <ID_1> wheels the cart out of the shaded lane and into the bustling square, slowing his push as the open space and distant crowds spread out ahead of him."),
        ],
    ),
    "1010": (
        "One person. <ID_1> a solo female hiker in her 30s, light skin, dark hair tucked under a knit cap, wearing a forest-green parka, amber-rimmed glasses, and a heavy gray wool scarf wound at her neck.",
        [
            ("A solo hiker steps off a packed dirt trail into a misty pine forest at early morning, her breath fogging as she moves between the tall trunks.",
             "Outdoors, a misty pine forest at early morning, soft diffuse light filtering through tall trunks; damp air and blurred undergrowth fill the muted background.",
             "Wide tracking shot, eye-level, shallow depth of field. The camera drifts backward, leading her into the trees. <ID_1> steps off the trail and walks into the misty forest, her breath fogging in the cold air as she picks her way between the towering pine trunks."),
            ("The hiker pushes onward as the trail descends into a hollow carpeted in soft brown pine needles, planting each step carefully on the slope.",
             "Outdoors, a shallow forest hollow carpeted with brown pine needles, dim green light beneath the dense canopy; blurred ferns and trunks ring the background.",
             "Medium tracking shot, eye-level, shallow depth of field. The camera tracks beside her down the gentle slope. <ID_1> picks her way carefully down the needle-covered descent, one hand brushing a trunk for balance, her boots sinking slightly into the soft forest floor."),
            ("The hiker comes upon a small clearing where a fallen log lies green with lichen and white mushrooms, crouching to look more closely.",
             "Outdoors, a small forest clearing with a mossy fallen log, green lichen and white mushrooms along its length; soft diffuse light falls through the canopy.",
             "Medium shot, eye-level, shallow depth of field. The camera holds steady as she lowers into frame. <ID_1> stops at the fallen log and crouches down to look, tilting her head and reaching out a careful hand toward the pale mushrooms growing along the mossy bark."),
            ("The hiker follows a thin stream snaking between mossy stones in a sun-dappled glade, crouching briefly to trail her fingers in the water.",
             "Outdoors, a sun-dappled glade with a thin stream over mossy stones, sparkling water and bright green moss; warm light breaks through the canopy above.",
             "Medium shot, eye-level, shallow depth of field. The camera tracks slowly along the streambed beside her. <ID_1> follows the thin stream over the mossy stones and crouches at its edge, trailing her fingers through the cold water as the dappled light shifts across her."),
            ("The hiker climbs stone steps cut into a hillside slick with damp moss, ferns brushing her sleeves as she steadies herself on the rise.",
             "Outdoors, stone steps cut into a mossy hillside, crowding ferns leaning over the path; cool damp shadow broken by faint shafts of filtered light.",
             "Medium tracking shot, low angle, shallow depth of field. The camera tilts up to follow her ascent. <ID_1> climbs the moss-slick stone steps, one hand pressed to the rock for balance, the crowding ferns brushing against her sleeves as she steadily works her way up."),
            ("The hiker crests the hillside into a ridgetop opening where the fog parts to reveal a sea of pines, stopping to gaze out over the valley.",
             "Outdoors, a ridgetop opening where fog parts over a vast sea of pine treetops; bright cool morning light spreads across the distant rolling slopes.",
             "Wide shot, eye-level, shallow depth of field. The camera holds steady as she rises into the open. <ID_1> crests the ridge and comes to a stop, her shoulders settling as she gazes out over the sea of pines, the parting fog drifting past her into the bright valley."),
        ],
    ),
}


def main():
    import re, statistics
    out = "/mnt/beegfs/xiangbo/helios_runs/eval_prompts_cfr/evtsw.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ev_w, bg_w, sc_w, ro_w = [], [], [], []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt_index", "prompt"])
        for vid, (role, segs) in DATA.items():
            ro_w.append(len(role.split()))
            for i, (event, bg, scene) in enumerate(segs):
                ev_w.append(len(event.split())); bg_w.append(len(bg.split())); sc_w.append(len(scene.split()))
                prompt = (
                    f"<header>{HEADER}</header> <event>{event}</event> <role>{role}</role> "
                    f"<Background>{bg}</Background> <style>{STYLE}</style> <scene>{scene}</scene>"
                )
                w.writerow([vid, i, prompt])
    n = sum(len(s) for _, s in DATA.values())
    print(f"wrote {out}: {len(DATA)} videos x 6 events = {n} rows (6-tag format)")
    print("--- per-tag word counts (mine  vs  training target) ---")
    print(f"event  mean={statistics.mean(ev_w):5.1f}  range[{min(ev_w)}-{max(ev_w)}]   target ~22 (15-32)")
    print(f"role   mean={statistics.mean(ro_w):5.1f}  range[{min(ro_w)}-{max(ro_w)}]   target ~27 (20-36)")
    print(f"bg     mean={statistics.mean(bg_w):5.1f}  range[{min(bg_w)}-{max(bg_w)}]   target ~23 (15-33)")
    print(f"scene  mean={statistics.mean(sc_w):5.1f}  range[{min(sc_w)}-{max(sc_w)}]   target ~39 (29-51)")
    e, b, s = DATA["1000"][1][0]
    print("--- sample (id=1000 idx0) ---")
    print(f"<header>{HEADER}</header> <event>{e}</event> <role>{DATA['1000'][0]}</role> <Background>{b}</Background> <style>{STYLE}</style> <scene>{s}</scene>")
    print("tags:", re.findall(r'<([a-zA-Z_]+)>', f"<header>{HEADER}</header> <event>{e}</event> <role>{DATA['1000'][0]}</role> <Background>{b}</Background> <style>{STYLE}</style> <scene>{s}</scene>"))


if __name__ == "__main__":
    main()
