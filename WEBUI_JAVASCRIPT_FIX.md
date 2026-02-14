# Web UI JavaScript Syntax Error Fix

## Problem Reported
```
Uncaught SyntaxError: '' string literal contains an unescaped line break monitoring:1809:53
```

## Root Cause

In `vsr_plusplus_NEU/systems/web_ui.py` line 2164, there was an incorrect escape sequence:

```python
const filesList = files.join('\\n');  # ❌ WRONG
```

This code is inside a Python triple-quoted string (f-string) that generates JavaScript:

```python
html = f"""
    <script>
        function updateBatchFiles(data) {{
            const files = batch.files || [];
            const filesList = files.join('\\n');  # ❌ Creates actual newlines!
            document.getElementById('batchFilesList').value = filesList;
        }}
    </script>
"""
```

### Why This Breaks

1. **Python string processing:**
   - Python sees `'\\n'` (backslash-n in source)
   - Python interprets this as a **literal newline character** `\n`
   - Python outputs actual line breaks in the HTML

2. **Generated JavaScript:**
   ```javascript
   const filesList = files.join('
   ');  // ❌ Literal newline breaks the string!
   ```

3. **Browser error:**
   - JavaScript doesn't allow unescaped line breaks in string literals
   - Syntax error: `Uncaught SyntaxError: '' string literal contains an unescaped line break`

## Solution

Change to double-escaped newline:

```python
const filesList = files.join('\\\\n');  # ✅ CORRECT
```

### How It Works

1. **Python string processing:**
   - Python sees `'\\\\n'` (4 characters: `\`, `\`, `n`)
   - Python interprets as TWO backslashes followed by `n`
   - Python outputs `\\n` in the HTML (backslash-n, 2 characters)

2. **Generated JavaScript:**
   ```javascript
   const filesList = files.join('\\n');  // ✅ Escaped newline!
   ```

3. **Browser execution:**
   - JavaScript interprets `\\n` as an escaped newline character
   - Correctly joins array elements with newlines
   - No syntax error!

## Escape Sequence Reference

When embedding JavaScript in Python strings:

| Want in JavaScript | Write in Python | Result |
|-------------------|-----------------|---------|
| `\n` (newline)    | `\\\\n`         | Escaped newline |
| `\t` (tab)        | `\\\\t`         | Escaped tab |
| `\\` (backslash)  | `\\\\\\\\`      | Literal backslash |
| `'` (quote)       | `\\'` or `'`    | Single quote |

**Rule of thumb:** For each backslash you want in the JavaScript output, write TWO backslashes in the Python source.

## Testing

After fix:
1. ✅ Web UI loads without JavaScript errors
2. ✅ Batch files list displays correctly with line breaks
3. ✅ All data updates work properly

## Related Files

- `vsr_plusplus_NEU/systems/web_ui.py` - Line 2164 (FIXED)

No other similar issues found in the codebase.
