# The test set: who, what photos, and their consent

Every keep-or-drop decision in this project is a score on real people's photos
(`score_run.py`). This file says how those photos are taken and kept. The
photos, the register and the set file live in `data/testset/`, which git
ignores; `premerge.py` fails if any of it is ever tracked. Only this template
is in the repository.

## Who

Adults only, each with a signed consent form (below) on file before the first
photo. Start with whoever can sit for it now; scores stay **provisional**
until ten people are in (`score_run.py --gate` says so on every verdict). As
people join, aim for a spread of skin tones, genders, hair (short, long,
curly, covered, none), beards and glasses: a change that only helps one kind
of face shows up as a loss on the others.

Give each person a pseudonym, `p01`, `p02`, ... Names go on the paper form
only, never in a file name or the set file.

## The photos, per person

Same session, same light, phone or webcam at eye height, about an arm's length
away, the face at least 500 pixels tall (250 is the floor), nothing over the
eyes or brows.

| File | What | Used for |
|---|---|---|
| `front.jpg` | straight at the camera, mouth closed, face relaxed | built from |
| `left35.jpg`, `right35.jpg` | head turned 30-40 degrees each way, chin level | built from (`sides`) |
| `left50.jpg`, `right50.jpg` | the same at about 50 degrees | fusion tests only |
| `left20.jpg` | turned about 20 degrees | **never built from**: judges the shape (`held_out`) |
| `o1.jpg`, `o2.jpg` | any two other photos, another day or another light | **never built from**: judges likeness (`others`) |
| `smile.jpg` | optional, a smile showing teeth | teeth checks |

The studio's guided capture takes `front`, `left35` and `right35` with its
checks; take the rest by hand. Put each person's photos in
`data/testset/pNN/`.

## The register: `data/testset/consent.csv`

```
consent_id,pseudonym,date,scopes,retain_until,withdrawn
c01,p01,2026-10-27,build;score,2027-10-27,
```

- `scopes`: what the person agreed to, `;`-separated. `build` (avatars made
  from the photos) and `score` (face-recognition similarity measured on them)
  are needed for the test set; `photo_check` only if they also agreed to the
  photos being sent to the Claude API.
- `retain_until`: the date the photos are deleted by. `score_run.py` leaves a
  face out from the day after.
- `withdrawn`: the date someone withdrew. Their face is left out at once;
  delete their folder, their built avatars (`out/*-pNN*`) and their renders
  (`out/renders/pNN-*`) the same day.

## The set: `data/testset/test-set.json`

```json
[{"name": "p01", "photo": "data/testset/p01/front.jpg", "gender": "male",
  "consent_id": "c01", "split": "dev",
  "sides": ["data/testset/p01/left35.jpg", "data/testset/p01/right35.jpg"],
  "held_out": "data/testset/p01/left20.jpg",
  "others": ["data/testset/p01/o1.jpg", "data/testset/p01/o2.jpg"],
  "tags": {"skin": "IV", "hair": "short", "beard": false, "glasses": false},
  "hair_cut": "hair_short01"}]
```

- `split`: `dev` for tuning; about one person in five goes to `lockbox`, which
  is scored only at sign-off (`--split lockbox`) so tuning cannot fit it.
- `hair_cut`: the catalogue cut a person would pick for them, for the haircut
  tests. Optional.

## Running it

```
python score_run.py p0 --noise                          # once: baseline and each score's spread
python score_run.py TAG --against p0 --gate others_p    # per change: exits 1 to drop it
```

Re-run `--noise` when people join.

## Consent form

Fill in one per person, on paper or as a signed PDF kept beside the register
(git-ignored). This is a plain-language template, not legal advice. Check it
against the law where you are before use; biometric data has its own rules in
many places (in the EU and UK it is special-category data under the GDPR).

> **Photos for testing the avatar engine**
>
> **What this is for.** We are testing software that makes a 3D avatar from
> photos. Your photos are used only to test and improve it, inside the team.
> They are not published, sold, or used to train anyone else's models.
>
> **What we take.** About ten photos of your head and shoulders from a few
> angles, in one session, and up to two other photos of you that you choose.
>
> **What we do with them.** We build avatars from them and measure how much
> each avatar looks like you, including with face-recognition software that
> turns a face into numbers (a biometric measurement). Those numbers are
> compared and then discarded with the avatar; they are not used to identify
> you anywhere else.
>
> **Where they are kept.** On one computer, `__________`, only. Not in any
> code repository, cloud storage or shared drive. Files carry a code
> (`p__`), not your name; only this form links the two.
>
> **How long.** Until `__________` (at most two years), then deleted.
>
> **Withdrawing.** You can withdraw at any time without giving a reason, by
> telling `__________`. Your photos and everything made from them are deleted
> within one working day.
>
> Please tick each you agree to:
>
> - [ ] I am 18 or older.
> - [ ] Photos of me may be taken and kept as described above.
> - [ ] Avatars may be made from them, for the team's testing only.
> - [ ] Face-recognition similarity (a biometric measurement) may be computed
>       on them, as described above.
> - [ ] *(Optional)* They may be sent to the Claude API (Anthropic) for an
>       automatic photo-quality check. Leave this unticked to keep them on the
>       computer above only.
>
> Name: `__________`  Code: `p__`  Signature: `__________`  Date: `__________`
