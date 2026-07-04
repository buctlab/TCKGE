# Load the JSON configuration
$configPath = ".\config.json"
$config = Get-Content $configPath | ConvertFrom-Json

# Define the folder and suffixes from the configuration
$folder = $config.folder
$suffix = $config.epssuffix
$pdfSuffix = $config.pdfSuffix

# Get the full path of the folder
$folderPath = Join-Path -Path (Get-Location) -ChildPath $folder

# Check if the folder exists
if (-Not (Test-Path $folderPath)) {
    Write-Host "The specified folder '$folder' does not exist."
    exit
}

# Get all files in the folder and filter for PNG and JPG
$images = Get-ChildItem -Path $folderPath -File -Force | Where-Object { $_.Extension -eq ".png" -or $_.Extension -eq ".jpg" }

# Debug output to check the number of images found
Write-Host "Found $($images.Count) images in '$folderPath'."

# Convert each image to EPS format using bmeps
foreach ($image in $images) {
    Write-Host "Processing image: $($image.FullName)"
    
    # Construct the EPS file name
    $epsFileName = [System.IO.Path]::GetFileNameWithoutExtension($image.Name) + $suffix + ".eps"
    $epsFilePath = Join-Path -Path $folderPath -ChildPath $epsFileName

    # Use bmeps tool to convert the image to EPS
    Write-Host "Converting to: $epsFilePath"
    try {
        & bmeps -c $image.FullName $epsFilePath
        Write-Host "Converted '$($image.Name)' to '$epsFileName'"
    } catch {
        Write-Host "Error converting '$($image.Name)': $_"
    }

    # Now convert the EPS file to PDF using epstopdf
    $pdfFileName = [System.IO.Path]::GetFileNameWithoutExtension($epsFileName) + $pdfSuffix + ".pdf"
    $pdfFilePath = Join-Path -Path $folderPath -ChildPath $pdfFileName

    Write-Host "Converting EPS to PDF: $pdfFilePath"
    try {
        & epstopdf $epsFilePath --outfile=$pdfFilePath
        Write-Host "Converted '$epsFileName' to '$pdfFileName'"
    } catch {
        Write-Host "Error converting '$epsFileName' to PDF: $_"
    }
}