# Random Artists and Tags Fork
![Screenshot of the plugin](Screenshot.png)
![Screenshot of the plugin](Screenshot2.png)

This fork adds functionality for randomly selecting artists and tags:
- Random artists button that pulls from:
  - e621_artist_webui.csv
  - danbooru_artist_webui.csv
- Random tags button that fetches tags from random posts on either site
- Customizable settings menu with:
  - Categories list for filtering
  - Unwanted tags exclusion list
- Favorite artists dropdown
- Solo-only checkbox, skip posts that don't fit this criteria

## Installation

1. Install the required Python dependencies (run PowerShell as administrator):
```powershell
python -m pip install beautifulsoup4 requests curl_cffi --target="C:\Program Files\Krita (x64)\lib\site-packages"
```

2. Download and install the correct cffi_backend package file manually (run in PowerShell as administrator):
```powershell
# Create a temporary directory
New-Item -ItemType Directory -Path "$env:TEMP\cffi_temp" -Force
Set-Location "$env:TEMP\cffi_temp"

# Download the wheel file using PowerShell's built-in web client
Invoke-WebRequest -Uri "https://files.pythonhosted.org/packages/d1/b6/0b0f5ab93b0df4acc49cae758c81fe4e5ef26c3ae2e10cc69249dfd8b3ab/cffi-1.17.1-cp310-cp310-win_amd64.whl" -OutFile "cffi.whl"

# Rename to zip and extract
Rename-Item -Path "cffi.whl" -NewName "cffi.zip"
Expand-Archive -Path "cffi.zip" -DestinationPath "."

# Copy the backend file to Krita python packages folder
Copy-Item -Path "_cffi_backend.cp310-win_amd64.pyd" -Destination "C:\Program Files\Krita (x64)\lib\krita-python-libs" -Force
```

**Note:** You must run PowerShell as administrator for these commands to work. These commands will automatically download and install the required CFFI backend file from Python's official package repository.

3. Extract the release to
```
%appdata%\krita\pykrita
```

## Usage

After installation, you'll see new buttons in the plugin interface:
- Random Artist: Selects a random artist from the included databases
- Random Tags: Fetches tags from random posts
- Settings: Configure filtering options and preferences
- Favorites: Quick access to your favorite artists
- Solo Only: Toggle to only include solo posts when fetching random tags

Configure your preferences in the Settings menu to customize the tag filtering and artist selection process.

## Troubleshooting

If you encounter permission errors:
- Make sure you're running PowerShell as administrator
- Check that Krita is not running during installation
- Verify the installation paths exist on your system

For other issues, please open an issue on the GitHub repository.
