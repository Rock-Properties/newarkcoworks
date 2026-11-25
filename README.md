# Newark Coworks

The website for Newark Coworks, the coworking space at 368 Broad Street, Suite 2, Newark, NJ: [newarkcoworks.com](https://newarkcoworks.com).

It is a plain static site: one HTML page, one stylesheet, no build step and no JavaScript. Open `index.html` in a browser to see it.

## What's here

| Path | What it is |
| --- | --- |
| `index.html` | The whole page: hero, the space, membership, scholarship, getting here, hosts, visit |
| `styles.css` | All styling. Colors and fonts are set once at the top under `:root` |
| `images/` | Web-ready photos (two sizes each) and `share-card.jpg` for link previews |
| `fonts/` | Besley (headings) and Libre Franklin (text), self-hosted, with their open-font licenses |
| `favicon.svg` | Browser tab icon |
| `robots.txt`, `sitemap.xml` | For search engines |
| `tools/build_images.py` | Rebuilds `images/` from original photos |

## Changing things

- **Price or opening date**: search `index.html` for `$150` and `January 1, 2027`. They appear in the page text, the pass in the hero, and the `<meta>` descriptions at the top.
- **Contact details**: search `index.html` for `mailto:` and `tel:` to find every one. They are also in the JSON block in `<head>`.
- **Membership checkout**: the "Become a member" button (`id="join"`) currently opens an email. When online payment is set up, replace its `href` with the checkout link and delete the line under it that says it opens an email.
- **Scholarship application**: the "Apply for a scholarship" button links to a Google Form. Replace its `href` to point somewhere else.
- **Photos**: put new originals in a folder and run

  ```
  pip install pillow
  python3 tools/build_images.py "/path/to/originals" images
  ```

  The crops are listed at the top of that script.

## Hosting

The site is served by GitHub Pages from the `main` branch, root folder, with `newarkcoworks.com` as the custom domain (Settings → Pages). Pushing to `main` updates the live site within a minute or two.

DNS for the domain:

| Type | Host | Value |
| --- | --- | --- |
| A | @ | 185.199.108.153 |
| A | @ | 185.199.109.153 |
| A | @ | 185.199.110.153 |
| A | @ | 185.199.111.153 |
| CNAME | www | `rock-properties.github.io` |
