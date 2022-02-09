# model using Windows kernel emulator

## modifications made to `speakeasy`

### Function `get_char_width` in `speakeasy/winenv/api/api.py`:
```
if name.endswith('A'):
    return 1
elif name.endswith('W'):
    return 2
raise ApiEmuError('Failed to get character width from function: %s' % (name))
```
Modified to assume ascii char width by default:
```
if name.endswith('W'):
    return 2
else:
    return 1
```
