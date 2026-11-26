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
| `images/logo-mark.webp` | The logo: the House of Prayer and the Plume House, drawn from a photo of Broad Street |
| `favicon-32.png`, `favicon-192.png`, `apple-touch-icon.png` | Browser tab and home-screen icons, made from the logo |
| `robots.txt`, `sitemap.xml` | For search engines |
| `tools/build_images.py` | Rebuilds `images/` from original photos |
| `tools/build_logo.py` | Redraws the logo from the photo and writes the mark, the icons and the logo files |

## Changing things

- **Price or opening date**: search `index.html` for `$150` and `January 1, 2027`. They appear in the page text, the pass in the hero, and the `<meta>` descriptions at the top.
- **Contact details**: search `index.html` for `mailto:` and `tel:` to find every one. They are also in the JSON block in `<head>`.
- **Membership application**: the three "Become a member" buttons open a Microsoft Form. Search `index.html` for `forms.cloud.microsoft` to find them and replace the `href` to send people somewhere else.
- **Scholarship application**: the "Apply for a scholarship" button opens its own Microsoft Form, found the same way.
- **Photos**: put new originals in a folder and run

  ```
  pip install pillow
  python3 tools/build_images.py "/path/to/originals" images
  ```

  The crops are listed at the top of that script.
- **Logo**: the drawing is made from one photo (`IMG_5567.jpeg` in the Newark Coworks Dropbox folder). The script cuts the two buildings out along their outline, paints out the poles and signs in front of them, fills each surface with its real color and traces the features in pencil. To redraw it, or to get the logo files again:

  ```
  pip install pillow numpy opencv-contrib-python
  python3 tools/build_logo.py "/path/to/IMG_5567.jpeg" . --files logo-files
  ```

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
