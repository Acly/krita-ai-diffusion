# Tag Autocompletion Data Files

This directory contains tag files used for autocompletion in AI Diffusion. By default we ship:

- `Danbooru.csv`: SFW tags from danbooru.donmai.us (anime)
- `Danbooru NSFW.csv`: NSFW tags from danbooru.donmai.us (anime)
- `e621.csv`: SFW tags from e621.net (furry)
- `e621 NSFW.csv`: NSFW tags from e621 (furry)

Each file is a comma-separated CSV with the columns "tag, type, count, aliases":

- `tag`: the actual tag. Underlines are stripped out on loading.
- `type`: the numeric type. Corresponds to Danbooru categories:
    general, artist, copyright, character, meta.
    This is used to color the tag entry.
- `count`: how often the tag is used on the site. Used to sort the tag list.
- `aliases`: other names for this tag. Not currently used.

The first line contains the column names.

# Truncation

Though they can understand general tags with very low example count thanks to their knowledge of English,
networks can rarely do useful things with small numbers of settings, artists or characters. As such,
we truncate general and meta tags with less than 20 occurrences, and setting, artist and character
tags with less than 50 occurrences, using this command:

```sh
awk -i inplace '
BEGIN { FS=OFS="," }
NR == 1 { print; next }  # Always print the header
{
    if (($2 == 0 || $2 == 5) && $3 >= 20) print;
    else if ($3 >= 50) print;
}
' ai_diffusion/tags/*.csv
```

# Generation

How these files were created.

## Danbooru

The Danbooru tags were generated from the Danbooru public tag dataset at
https://console.cloud.google.com/bigquery?project=danbooru1.

A SFW tag is defined as a tag that only applies on SFW (`rating:general`) images.
A NSFW tag is any other tag.

### Danbooru.csv

Run the following SQL in Google BigQuery, then save the result as "Danbooru.csv":

```sql
-- SFW Tags Query
WITH sfw_posts AS (
  SELECT *
  FROM `danbooru1.danbooru_public.posts`
  WHERE rating = 'g'
),
sfw_tags AS (
  SELECT DISTINCT tag
  FROM sfw_posts,
  UNNEST(SPLIT(tag_string, ' ')) AS tag
),
tag_info AS (
  SELECT
    t.name AS tag,
    t.category AS type,
    t.post_count AS count,
    STRING_AGG(CASE WHEN ta.status = 'active' THEN ta.antecedent_name ELSE NULL END, ',') AS aliases
  FROM `danbooru1.danbooru_public.tags` t
  LEFT JOIN `danbooru1.danbooru_public.tag_aliases` ta
    ON t.name = ta.consequent_name
  GROUP BY t.name, t.category, t.post_count
)
SELECT
  ti.tag,
  ti.type,
  ti.count,
  IFNULL(ti.aliases, '') AS aliases
FROM tag_info ti
INNER JOIN sfw_tags st ON ti.tag = st.tag
ORDER BY ti.count DESC
LIMIT 100000;
```

### Danbooru NSFW.csv


Run the following SQL in Google BigQuery, then save the result as "Danbooru NSFW.csv":

```sql
-- NSFW Tags Query
WITH sfw_posts AS (
  SELECT *
  FROM `danbooru1.danbooru_public.posts`
  WHERE rating = 'g'
),
sfw_tags AS (
  SELECT DISTINCT tag
  FROM sfw_posts,
  UNNEST(SPLIT(tag_string, ' ')) AS tag
),
all_used_tags AS (
  SELECT DISTINCT tag
  FROM `danbooru1.danbooru_public.posts`,
  UNNEST(SPLIT(tag_string, ' ')) AS tag
),
tag_info AS (
  SELECT
    t.name AS tag,
    t.category AS type,
    t.post_count AS count,
    STRING_AGG(CASE WHEN ta.status = 'active' THEN ta.antecedent_name ELSE NULL END, ',') AS aliases
  FROM `danbooru1.danbooru_public.tags` t
  LEFT JOIN `danbooru1.danbooru_public.tag_aliases` ta
    ON t.name = ta.consequent_name
  GROUP BY t.name, t.category, t.post_count
)
SELECT
  ti.tag,
  ti.type,
  ti.count,
  IFNULL(ti.aliases, '') AS aliases
FROM tag_info ti
WHERE ti.tag NOT IN (SELECT tag FROM sfw_tags)
  AND ti.tag IN (SELECT tag FROM all_used_tags)
ORDER BY ti.count DESC
LIMIT 100000;
```

## e621

The e621 tags were generated from the e621 public tag dataset at https://e621.net/db_export/.

The NSFW split is a bit more complicated here: there are some tags that appear overwhelmingly
on NSFW posts but occasionally do show up on SFW ones. We filter those out by defining tags
that appear 99% on NSFW posts as NSFW. We also remap the e621 categories to Danbooru.

For processing efficiency, we temporarily import the dataset into a sqlite3 database:

```sh
pv posts-2024-08-07.csv.gz |gunzip - |\
  sqlite3 e621.db -cmd ".mode csv" ".import /dev/stdin posts"
pv tags-2024-08-07.csv.gz | gunzip - |\
  sqlite3 e621.db -cmd ".mode csv" ".import /dev/stdin tags"
pv tag_aliases-2024-08-07.csv.gz | gunzip - |\
  sqlite3 e621.db -cmd ".mode csv" ".import /dev/stdin tag_aliases"
```

Make sure you have tqdm installed:

```sh
pip3 install tqdm
```

Then run this script:

```python
import sqlite3
import csv
from tqdm import tqdm
from collections import Counter

# tags that are 99% used in NSFW images are NSFW, even if they have some SFW use.
NSFW_RATIO = 0.99

def split_tags(tag_string):
    return set(tag_string.split())

def remap_category(category):
    category = int(category)
    if category in [5, 8]:  # SPECIES and LORE
        return 0  # Map to GENERAL
    elif category == 7:  # META
        return 5  # Map to META
    elif category == 6:  # INVALID
        return None  # We'll skip these tags
    else:
        return category

# Connect to the SQLite database
conn = sqlite3.connect('e621.db')
cursor = conn.cursor()

# Get total number of non-deleted posts
cursor.execute("SELECT COUNT(*) FROM posts WHERE is_deleted = 'f'")
total_posts = cursor.fetchone()[0]

# Count tag occurrences in SFW and NSFW posts
print("Counting tag occurrences...")
sfw_tag_counts = Counter()
nsfw_tag_counts = Counter()

cursor.execute("SELECT tag_string, rating FROM posts WHERE is_deleted = 'f'")
for tag_string, rating in tqdm(cursor, total=total_posts):
    tags = split_tags(tag_string)
    if rating == 's':
        sfw_tag_counts.update(tags)
    else:
        nsfw_tag_counts.update(tags)

# Determine predominantly NSFW tags
predominantly_nsfw_tags = set()
for tag in sfw_tag_counts:
    total_count = sfw_tag_counts[tag] + nsfw_tag_counts[tag]
    if nsfw_tag_counts[tag] / total_count > NSFW_RATIO:
        # print(f"Drop {tag} from SFW set as it is predominantly NSFW.")
        predominantly_nsfw_tags.add(tag)

# Get SFW tags (excluding predominantly NSFW tags)
sfw_tags = set(sfw_tag_counts.keys()) - predominantly_nsfw_tags

# Get all used tags
all_used_tags = set(sfw_tag_counts.keys()) | set(nsfw_tag_counts.keys())

# Get NSFW tags
nsfw_tags = all_used_tags - sfw_tags

# Prepare tag info
print("Fetching tag info...")
cursor.execute("""
    SELECT t.name, t.category, t.post_count,
           GROUP_CONCAT(ta.antecedent_name, ',') as aliases
    FROM tags t
    LEFT JOIN tag_aliases ta ON t.name = ta.consequent_name AND ta.status = 'active'
    GROUP BY t.name
""")
tag_info = {}
for row in cursor.fetchall():
    remapped_category = remap_category(row[1])
    if remapped_category is not None:
        tag_info[row[0]] = {
            'category': remapped_category,
            'post_count': int(row[2]),
            'aliases': row[3] or ''
        }

# Sort and limit SFW tags
print("Sorting SFW tags...")
sorted_sfw_tags = sorted(
    [tag for tag in sfw_tags if tag in tag_info],
    key=lambda x: tag_info[x]['post_count'],
    reverse=True
)[:100000]

# Sort and limit NSFW tags
print("Sorting NSFW tags...")
sorted_nsfw_tags = sorted(
    [tag for tag in nsfw_tags if tag in tag_info],
    key=lambda x: tag_info[x]['post_count'],
    reverse=True
)[:100000]

# Write SFW tags to file
print("Writing SFW tags...")
with open('e621.csv', 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['tag', 'type', 'count', 'aliases'])
    for tag in tqdm(sorted_sfw_tags):
        info = tag_info[tag]
        writer.writerow([tag, info['category'], info['post_count'], info['aliases']])

# Write NSFW tags to file
print("Writing NSFW tags...")
with open('e621 NSFW.csv', 'w', newline='') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['tag', 'type', 'count', 'aliases'])
    for tag in tqdm(sorted_nsfw_tags):
        info = tag_info[tag]
        writer.writerow([tag, info['category'], info['post_count'], info['aliases']])

conn.close()
print("Processing complete. Check e621.csv and e621 NSFW.csv for results.")
```
