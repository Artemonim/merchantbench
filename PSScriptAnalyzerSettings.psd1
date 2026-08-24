@{
    Severity     = @('Error', 'Warning')
    ExcludeRules = @(
        # * CI writes colored one-line stage status to the console; Write-Host is the intended API.
        'PSAvoidUsingWriteHost'
        # * Internal helpers write cache/report files; they are not user-facing cmdlets.
        'PSUseShouldProcessForStateChangingFunctions'
        # * Stage result/issue collectors use plural nouns by design (AE2 naming).
        'PSUseSingularNouns'
        # * Repo sources are UTF-8 without BOM (PS 7 utf8 encoding is BOM-less).
        'PSUseBOMForUnicodeEncodedFile'
    )
}
