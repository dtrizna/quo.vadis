# model using Windows kernel emulator

## Interesting fields besides `apis`:

```
file_access              [{'event': 'create', 'path': 'C:\Users\speakea...
registry_access          [{'event': 'open_key', 'path': 'HKEY_CURRENT_U...
process_events           [{'event': 'create', 'pid': 1524, 'path': 'C:\...
dynamic_code_segments                                                   []
dropped_files            [{'path': 'C:\Users\speakeasy_user\dgwgkEwU\Ok...
```

`file_access`:
```
[{'event': 'create', 'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM', 'open_flags': ['OPEN_EXISTING'], 'access_flags': ['GENERIC_READ', 'GENERIC_WRITE']}, {'event': 'create', 'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM', 'open_flags': ['OPEN_ALWAYS'], 'access_flags': ['GENERIC_READ', 'GENERIC_WRITE']}, {'event': 'create', 'path': 'C:\\ProgramData\\IyAoAoIQ\\WmgAAUEI', 'open_flags': ['OPEN_EXISTING'], 'access_flags': ['GENERIC_READ', 'GENERIC_WRITE']}, {'event': 'create', 'path': 'C:\\ProgramData\\IyAoAoIQ\\WmgAAUEI', 'open_flags': ['OPEN_ALWAYS'], 'access_flags': ['GENERIC_READ', 'GENERIC_WRITE']}, {'event': 'create', 'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM.exe', 'open_flags': ['CREATE_ALWAYS'], 'access_flags': ['GENERIC_READ', 'GENERIC_WRITE']}, {'event': 'write', 'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM.exe', 'data': 'TVqQAAMAAAA...AAAA=', 'size': 203776, 'buffer': '0x52000'}]
```

`registry_access`:
```
{'event': 'open_key', 'path': 'HKEY_CURRENT_USER\\software\\microsoft\\windows\\currentversion\\run'}]
```

`dropped_files`:
```
[{'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM', 'size': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, {'path': 'C:\\ProgramData\\IyAoAoIQ\\WmgAAUEI', 'size': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}, {'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM.exe', 'size': 203776, 'sha256': '088273547b0f59926ad67619fbc659e64138bc191cfeffe9853c912dd56a7187'}]
```

`process_events`:
```
[{'event': 'create', 'pid': 1524, 'path': 'C:\\Users\\speakeasy_user\\dgwgkEwU\\OkIkcgIM.exe', 'cmdline': ''}]
```

## Common errors:

- unsupported API: `error.api_name` contains API that is unsupported
- invalid_read: `error.instr` contains actual assembly instruction that raised an invalid read, e.g: `add byte ptr [eax], ah`


## modifications made to `speakeasy` (NOT UP TO DATE, some files has more APIs added)

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

### Added dummy API processors to `speakeasy/winenv/api/usermode/advapi32.py`:

```
    @apihook("InitializeSecurityDescriptor", argc=2, conv=_arch.CALL_CONV_STDCALL)
    def InitializeSecurityDescriptor(self, emu, argv, ctx={}):
        '''
        BOOL InitializeSecurityDescriptor(
            [out] PSECURITY_DESCRIPTOR pSecurityDescriptor,
            [in]  DWORD                dwRevision
        );
        '''
        return 0
    
    @apihook("SetSecurityDescriptorDacl", argc=4, conv=_arch.CALL_CONV_STDCALL)
    def SetSecurityDescriptorDacl(self, emu, argv, ctx={}):
        '''
        BOOL SetSecurityDescriptorDacl(
            [in, out]      PSECURITY_DESCRIPTOR pSecurityDescriptor,
            [in]           BOOL                 bDaclPresent,
            [in, optional] PACL                 pDacl,
            [in]           BOOL                 bDaclDefaulted
        );
        '''
        return 0

    @apihook("RegSetValueExW", argc=5, conv=_arch.CALL_CONV_STDCALL)
    def RegSetValueExW(self, emu, argv, ctx={}):
        '''
        LSTATUS RegSetValueExW(
            [in]           HKEY       hKey,
            [in, optional] LPCWSTR    lpValueName,
                            DWORD      Reserved,
            [in]           DWORD      dwType,
            [in]           const BYTE *lpData,
            [in]           DWORD      cbData
        );
        '''
        return 0
    
    ...
```

### Added dummy API processors to `speakeasy/winenv/api/usermode/kernel32.py`:

```
    @apihook("SetFileAttributesW", argc=2)
    def SetFileAttributesW(self, emu, argv, ctx={}):
        '''
        BOOL SetFileAttributesW(
            [in] LPCWSTR lpFileName,
            [in] DWORD   dwFileAttributes
        );
        '''
        return 0
    
    ...
```

### Added dummy API processors to `speakeasy/winenv/api/usermode/user32.py`:

```
    @apihook("GetMessageTime", argc=0)
    def GetMessageTime(self, emu, argv, ctx={}):
        '''
        LONG GetMessageTime();
        '''
        import time
        return int(time.time())
    
    ...
```

### API processor to `speakeasy/winenv/api/usermode/shell32.py`:

```
    @apihook('ordinal_680', argc=0)
    def ordinal_680(self, emu, argv, ctx={}):
        """
        BOOL IsUserAnAdmin();
        """
        return emu.get_user().get('is_admin', False)
```

### API processor to `speakeasy/winenv/api/usermode/winmm.py`:

```
    @apihook('timeGetDevCaps', argc=2)
    def timeGetDevCaps(self, emu, argv, ctx={}):        
        '''
        MMRESULT timeGetDevCaps(
            LPTIMECAPS ptc,
            UINT       cbtc
        );
        '''
        return 0 # MMSYSERR_NOERROR = 0
```
