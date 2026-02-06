/* tslint:disable */
/* eslint-disable */

export function receive_dropped_file(name: string, data: Uint8Array): void;

export function wasm_main(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly receive_dropped_file: (a: number, b: number, c: number, d: number) => void;
    readonly wasm_main: () => void;
    readonly main: (a: number, b: number) => number;
    readonly wasm_bindgen__closure__destroy__hb6ddae4ba5df266d: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h0254190e0f71d98f: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h96130d55a1194ed6: (a: number, b: number) => void;
    readonly wasm_bindgen__closure__destroy__h7f11d9994324913b: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h2a31e0ab946ab7e2: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h520cda017d487623: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h26368f60d434aa6e: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h269f14e85897dc46: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h27b8caa877d00a77: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h64940aa0d3860b54: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hcfc1639fda6ace14: (a: number, b: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
