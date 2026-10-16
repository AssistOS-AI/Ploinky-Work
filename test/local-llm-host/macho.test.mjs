// The strict library check of macOS (lib/local-llm-host/macho.mjs): the parsing of `otool -L` and `otool -l` (output captured from
// llama-b11295-bin-macos-arm64) and the resolution of a dependency inside a runnable copy. `otool` is injected, so these run on Linux too;
// the real check runs in the runner-lock-check workflow and on a Mac.
import '../home.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import test from 'node:test';

import { findMachoFiles, parseOtoolL, parseOtoolLoadCommands, resolveDependency, strictLibraryCheck, unresolvedDependencies } from '../../lib/local-llm-host/macho.mjs';
import { scratch } from './fixtures/engine.mjs';

const DYLIB_L = `/x/run/libggml-cpu.0.25.3.dylib:
\t@rpath/libggml-cpu.0.dylib (compatibility version 0.0.0, current version 0.25.3)
\t@rpath/libggml-base.0.dylib (compatibility version 0.0.0, current version 0.25.3)
\t/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate (compatibility version 1.0.0, current version 4.0.0)
\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 2100.43.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1356.0.0)
`;
const DYLIB_l = `/x/run/libggml-cpu.0.25.3.dylib:
Mach header
Load command 4
          cmd LC_ID_DYLIB
      cmdsize 56
         name @rpath/libggml-cpu.0.dylib (offset 24)
   time stamp 1 Thu Jan  1 02:00:01 1970
Load command 16
          cmd LC_RPATH
      cmdsize 32
         path @loader_path (offset 12)
Load command 17
      cmd LC_FUNCTION_STARTS
`;
const SERVER_L = `/x/run/llama-server:
\t/usr/lib/librdma.dylib (compatibility version 1.0.0, current version 1.0.0, weak)
\t@rpath/libllama.0.dylib (compatibility version 0.0.0, current version 0.0.0)
\t/opt/homebrew/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
`;
const SERVER_l = `/x/run/llama-server:
Load command 20
          cmd LC_RPATH
      cmdsize 32
         path @loader_path (offset 12)
`;

test('otool -L is parsed into its libraries, weak ones included, and otool -l into its rpaths and install name', () => {
    assert.deepEqual(parseOtoolL(DYLIB_L), [{ arch: null, libraries: ['@rpath/libggml-cpu.0.dylib', '@rpath/libggml-base.0.dylib', '/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate', '/usr/lib/libc++.1.dylib', '/usr/lib/libSystem.B.dylib'], unreadable: [] }]);
    assert.deepEqual(parseOtoolL(SERVER_L)[0].libraries, ['/usr/lib/librdma.dylib', '@rpath/libllama.0.dylib', '/opt/homebrew/opt/libomp/lib/libomp.dylib']);
    assert.deepEqual(parseOtoolLoadCommands(DYLIB_l), [{ arch: null, rpaths: ['@loader_path'], installName: '@rpath/libggml-cpu.0.dylib' }]);
    assert.deepEqual(parseOtoolLoadCommands(SERVER_l), [{ arch: null, rpaths: ['@loader_path'], installName: null }]);
    const fat = '/x/fat (architecture x86_64):\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n/x/fat (architecture arm64):\n\t@rpath/a.dylib (compatibility version 1.0.0, current version 1.0.0)\n';
    assert.deepEqual(parseOtoolL(fat).map((slice) => [slice.arch, slice.libraries]), [['x86_64', ['/usr/lib/libSystem.B.dylib']], ['arm64', ['@rpath/a.dylib']]]);
});

function tree(t) {
    const root = fs.realpathSync(scratch(t));
    const thin = Buffer.concat([Buffer.from('cffaedfe', 'hex'), Buffer.from('rest of the file')]);
    fs.writeFileSync(path.join(root, 'llama-server'), thin, { mode: 0o755 });
    fs.writeFileSync(path.join(root, 'libllama.0.5.0.dylib'), thin);
    fs.symlinkSync('libllama.0.5.0.dylib', path.join(root, 'libllama.0.dylib'));
    return root;
}

test('Mach-O files are found by their magic (thin and fat), whatever their name; links, text and Java classes are not', (t) => {
    const root = tree(t);
    fs.writeFileSync(path.join(root, 'be64'), Buffer.concat([Buffer.from('feedfacf', 'hex'), Buffer.alloc(8)]));
    fs.writeFileSync(path.join(root, 'fat'), Buffer.concat([Buffer.from('cafebabe', 'hex'), Buffer.from([0, 0, 0, 2]), Buffer.alloc(8)]));
    fs.writeFileSync(path.join(root, 'fat-le'), Buffer.concat([Buffer.from('bebafeca', 'hex'), Buffer.from([2, 0, 0, 0]), Buffer.alloc(8)]));
    fs.writeFileSync(path.join(root, 'Some.class'), Buffer.concat([Buffer.from('cafebabe', 'hex'), Buffer.from([0, 0, 0, 65]), Buffer.alloc(8)]));
    fs.writeFileSync(path.join(root, 'llama.txt'), 'plain text');
    fs.writeFileSync(path.join(root, 'tiny'), Buffer.from('cf'));
    fs.mkdirSync(path.join(root, 'sub'));
    fs.writeFileSync(path.join(root, 'sub', 'deep'), Buffer.concat([Buffer.from('cffaedfe', 'hex'), Buffer.alloc(8)]));
    assert.deepEqual(findMachoFiles(root).map((file) => path.relative(root, file)), ['be64', 'fat', 'fat-le', 'libllama.0.5.0.dylib', 'llama-server', path.join('sub', 'deep')]);
});

test('a dependency resolves through the file\'s rpaths, @loader_path and @executable_path to a file inside the copy, and never outside', (t) => {
    const root = tree(t);
    fs.mkdirSync(path.join(root, 'lib'));
    fs.writeFileSync(path.join(root, 'lib', 'libx.dylib'), 'x');
    const outside = fs.realpathSync(scratch(t));
    fs.writeFileSync(path.join(outside, 'libout.dylib'), 'x');
    fs.symlinkSync(path.join(outside, 'libout.dylib'), path.join(root, 'libabs.dylib'));
    const file = path.join(root, 'sub', 'tool');
    const base = { file, rpaths: ['@loader_path/..', '@executable_path/lib'], root, executableDir: root };
    assert.equal(resolveDependency('@rpath/libllama.0.dylib', base), path.join(root, 'libllama.0.5.0.dylib'), 'through @loader_path/.. and a relative link');
    assert.equal(resolveDependency('@rpath/libx.dylib', base), path.join(root, 'lib', 'libx.dylib'), 'through @executable_path');
    assert.equal(resolveDependency('@loader_path/../libllama.0.dylib', base), path.join(root, 'libllama.0.5.0.dylib'));
    assert.equal(resolveDependency('@executable_path/lib/libx.dylib', base), path.join(root, 'lib', 'libx.dylib'));
    assert.equal(resolveDependency('@rpath/libnope.dylib', base), null, 'an unresolved @rpath');
    assert.equal(resolveDependency('@rpath/libabs.dylib', base), null, 'a link to a file outside the copy');
    assert.equal(resolveDependency('@rpath/libllama.0.dylib', { ...base, rpaths: ['/opt/homebrew/lib'] }), null, 'an absolute rpath is not the copy');
    assert.equal(resolveDependency('/opt/homebrew/lib/libomp.dylib', base), null);
});

test('system libraries pass, the install name is not a dependency, and any other library fails naming the file and the library', (t) => {
    const root = tree(t);
    fs.writeFileSync(path.join(root, 'libggml-base.0.dylib'), 'x');
    fs.writeFileSync(path.join(root, 'libggml-cpu.0.dylib'), 'x');
    const dylib = unresolvedDependencies({ file: path.join(root, 'libggml-cpu.0.dylib'), listing: DYLIB_L, loadCommands: DYLIB_l, root, executableDir: root });
    assert.deepEqual(dylib, { unresolved: [] });
    const server = unresolvedDependencies({ file: path.join(root, 'llama-server'), listing: SERVER_L, loadCommands: SERVER_l, root, executableDir: root });
    assert.deepEqual(server, { unresolved: ['/opt/homebrew/opt/libomp/lib/libomp.dylib'] });
    assert.ok(unresolvedDependencies({ file: 'x', listing: '', loadCommands: '', root, executableDir: root }).error);
    const arch = unresolvedDependencies({ file: path.join(root, 'fat'), root, executableDir: root,
        listing: 'f (architecture arm64):\n\t/usr/local/lib/libz.dylib (compatibility version 1.0.0, current version 1.0.0)\n',
        loadCommands: 'f (architecture arm64):\nLoad command 0\n' });
    assert.deepEqual(arch, { unresolved: ['arm64: /usr/local/lib/libz.dylib'] });
});

test('a listing that cannot be read fails the file: an indented line that is no dependency, and a slice without any library', (t) => {
    const root = tree(t);
    const listing = (body) => `/x/llama-server:\n${body}`;
    const garbled = unresolvedDependencies({ file: path.join(root, 'llama-server'), root, executableDir: root, loadCommands: SERVER_l,
        listing: listing('\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n\t/opt/homebrew/lib/libz.dylib (version 1)\n') });
    assert.match(garbled.error, /not a dependency.*libz\.dylib/);
    assert.deepEqual(parseOtoolL(listing('\tgarbage\n'))[0].unreadable, ['garbage']);
    const empty = unresolvedDependencies({ file: path.join(root, 'llama-server'), root, executableDir: root, loadCommands: SERVER_l, listing: listing('') });
    assert.match(empty.error, /listed no library/);
    const bad = strictLibraryCheck(root, { otool: (flag) => ({ output: flag === '-L' ? listing('\t/opt/homebrew/lib/libz.dylib (version 1)\n') : SERVER_l }) });
    assert.equal(bad.ok, false, 'an unreadable dependency line fails the check');
    assert.match(bad.problems.join(), /otool could not check/);
    const none = strictLibraryCheck(root, { otool: (flag) => ({ output: flag === '-L' ? listing('') : SERVER_l }) });
    assert.equal(none.ok, false, 'a Mach-O file with no listed library fails the check');
});

test('the 64-bit fat magic is a Mach-O magic too', (t) => {
    const root = fs.realpathSync(scratch(t));
    fs.writeFileSync(path.join(root, 'fat64'), Buffer.concat([Buffer.from('cafebabf', 'hex'), Buffer.from([0, 0, 0, 1]), Buffer.alloc(8)]));
    assert.deepEqual(findMachoFiles(root).map((file) => path.basename(file)), ['fat64']);
});

test('the strict check passes a complete copy, fails an unresolved library and a file otool cannot read, and an empty copy', (t) => {
    const root = tree(t);
    const answers = (table) => (flag, file) => {
        const name = path.basename(file);
        if (table[name]?.error) return { error: table[name].error };
        return { output: flag === '-L' ? (table[name]?.L ?? SERVER_L.replace('/opt/homebrew/opt/libomp/lib/libomp.dylib', '/usr/lib/libc++.1.dylib')) : (table[name]?.l ?? SERVER_l) };
    };
    const clean = strictLibraryCheck(root, { otool: answers({ 'libllama.0.5.0.dylib': { L: `x:\n\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1.0.0)\n` } }), executable: 'llama-server' });
    assert.equal(clean.ok, true, clean.problems.join());
    assert.equal(clean.checked, 2);
    const defaults = strictLibraryCheck(root, { otool: answers({}), executable: 'llama-server' });
    assert.equal(defaults.ok, true, 'with the default listings both files need only system libraries and the copy\'s own');
    fs.unlinkSync(path.join(root, 'libllama.0.dylib'));
    const missing = strictLibraryCheck(root, { otool: answers({}) });
    assert.equal(missing.ok, false);
    assert.deepEqual(missing.unresolved, { 'libllama.0.5.0.dylib': ['@rpath/libllama.0.dylib'], 'llama-server': ['@rpath/libllama.0.dylib'] });
    assert.match(missing.problems.join(), /llama-server \(@rpath\/libllama\.0\.dylib\)/);
    const broken = strictLibraryCheck(root, { otool: answers({ 'llama-server': { error: 'ENOENT' } }) });
    assert.equal(broken.ok, false, 'a Mach-O file otool could not read is not a pass');
    assert.match(broken.problems.join(), /otool could not check/);
    const emptyDir = scratch(t);
    fs.writeFileSync(path.join(emptyDir, 'README'), 'nothing');
    assert.equal(strictLibraryCheck(emptyDir, { otool: answers({}) }).ok, false, 'a copy without any Mach-O file proves nothing');
});

test('negative control: a launched program\'s foreign library is named even when a same-named file exists elsewhere in the copy', (t) => {
    const root = tree(t);
    fs.writeFileSync(path.join(root, 'libomp.dylib'), 'x');
    const result = strictLibraryCheck(root, { otool: (flag) => ({ output: flag === '-L' ? SERVER_L : SERVER_l }), executable: 'llama-server' });
    assert.equal(result.ok, false);
    assert.match(result.problems.join(), /\/opt\/homebrew\/opt\/libomp\/lib\/libomp\.dylib/);
});
