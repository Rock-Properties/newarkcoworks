# Newark Coworks

The website for Newark Coworks, the coworking space inside Rock Properties' main office at 368 Broad Street, Suite 2, Newark, NJ. It will live at **newarkcoworks.com**.

It is a plain static site: one HTML page, one stylesheet, no build step and no JavaScript. Open `index.html` in a browser to see it.

## What's here

| Path | What it is |
| --- | --- |
| `index.html` | The whole page: hero, the space, what's included, getting here, hosts, book a tour |
| `styles.css` | All styling. Colors and fonts are set once at the top under `:root` |
| `images/` | Web-ready photos (two sizes each) and `share-card.jpg` for link previews |
| `fonts/` | Besley (headings) and Libre Franklin (text), self-hosted, with their open-font licenses |
| `favicon.svg` | Browser tab icon |
| `robots.txt`, `sitemap.xml` | For search engines |
| `tools/build_images.py` | Rebuilds `images/` from the original photos |

## Where the facts come from

Everything on the page comes from the **Newark Coworking** Asana project or the photos in the `Newark Coworks` Dropbox folder.

| On the site | Source |
| --- | --- |
| $150 a month, dedicated desk | Asana: "Price", "Write the Co-working Contract (lease)" |
| Opens January 1, 2027 | Asana section "Launch Date: Jan 1 2027" |
| Sit-to-stand desks | Asana: "Order 5 sit-to-stand desks" |
| Key fob and door code | Asana: "Get some more key fobs", "Program lock with some extra codes" |
| Windows that open | Asana: "Modify windows so that we can open" |
| CO2 monitor | Asana: "Set up the CO2 Detector" |
| Seltzer, nuts and chocolate each month | Asana: "Monthly drinks and snacks" |
| Coworking agreement | Asana: "Write the Co-working Contract (lease)" |
| Conference room, kitchenette, glass table, café table | The photos |

Not on the site because it isn't decided or written down yet: Wi-Fi details, access hours, parking, rooftop access, how many desks there are, and how members pay.

## Changing things

- **Price or opening date**: search `index.html` for `$150` and `January 1, 2027`. Each appears in the page text, the pass in the hero, and the `<meta>` descriptions at the top.
- **Contact details**: the tour email and the office phone appear in the "Book a tour" section, the footer, and the JSON block in `<head>`. Search `index.html` for `mailto:` and `tel:` to find every one.
- **Photos**: put new originals in a folder and run

  ```
  pip install pillow
  python3 tools/build_images.py "/path/to/Newark Coworks" images
  ```

  The crops are listed at the top of that script. The hero shot softly blurs one colleague's face.

## Putting it on newarkcoworks.com (GitHub Pages)

1. In this repository on GitHub: **Settings → Pages → Build and deployment**. Set Source to "Deploy from a branch", branch `main`, folder `/ (root)`, and save.
   GitHub Pages on a private repository needs a paid GitHub plan. On a free plan, make the repository public first (Settings → General → Change visibility).
2. Still under **Settings → Pages**, type `newarkcoworks.com` in **Custom domain** and save. Do this before the DNS step.
3. At the domain registrar, add these DNS records:

   | Type | Host | Value |
   | --- | --- | --- |
   | A | @ | 185.199.108.153 |
   | A | @ | 185.199.109.153 |
   | A | @ | 185.199.110.153 |
   | A | @ | 185.199.111.153 |
   | CNAME | www | `rock-properties.github.io` |

4. Back in **Settings → Pages**, tick **Enforce HTTPS** once the certificate is ready (DNS changes can take up to a day).

Any static host works just as well (Netlify, Cloudflare Pages, Vercel): point it at this repository with no build command and the repository root as the publish folder. Those can deploy from a private repository on their free plans.
