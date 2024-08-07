# Tag Autocompletion Data Files

This directory contains tag files used for autocompletion in AI Diffusion. By default we ship:

- `Danbooru.csv`: SFW tags from danbooru.donmai.us (anime)
- `Danbooru NSFW.csv`: NSFW tags from danbooru.donmai.us (anime)

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
