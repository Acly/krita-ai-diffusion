# Tag Autocompletion Data Files

This directory contains tag files used for autocompletion in AI Diffusion. By default we ship:

- `Danbooru.csv`: Tags from danbooru.donmai.us (anime)

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
