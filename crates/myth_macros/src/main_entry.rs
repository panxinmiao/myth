use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, ReturnType, Type};

pub fn generate(mut function: ItemFn) -> syn::Result<TokenStream> {
    if !function.sig.inputs.is_empty() {
        return Err(syn::Error::new_spanned(
            &function.sig.inputs,
            "#[myth::main] does not accept function arguments",
        ));
    }

    if function.sig.constness.is_some() {
        return Err(syn::Error::new_spanned(
            &function.sig.constness,
            "#[myth::main] cannot be applied to const functions",
        ));
    }

    let user_ident = format_ident!("__myth_user_main");
    function.sig.ident = user_ident.clone();

    let is_async = function.sig.asyncness.is_some();
    let return_type = function.sig.output.clone();
    let native_wrapper = native_wrapper(&user_ident, &return_type, is_async);
    let wasm_wrapper = wasm_wrapper(&user_ident, &return_type, is_async);

    Ok(quote! {
        #function

        #native_wrapper

        #wasm_wrapper

        #[cfg(target_arch = "wasm32")]
        fn main() {}
    })
}

fn native_wrapper(user_ident: &syn::Ident, return_type: &ReturnType, is_async: bool) -> TokenStream {
    match (is_async, return_type) {
        (false, ReturnType::Default) => quote! {
            #[cfg(not(target_arch = "wasm32"))]
            fn main() {
                let _ = ::myth::app::__macro_support::env_logger::try_init();
                #user_ident();
            }
        },
        (false, ReturnType::Type(_, ty)) => quote! {
            #[cfg(not(target_arch = "wasm32"))]
            fn main() -> #ty {
                let _ = ::myth::app::__macro_support::env_logger::try_init();
                #user_ident()
            }
        },
        (true, ReturnType::Default) => quote! {
            #[cfg(not(target_arch = "wasm32"))]
            fn main() {
                let _ = ::myth::app::__macro_support::env_logger::try_init();
                ::myth::app::__macro_support::pollster::block_on(#user_ident());
            }
        },
        (true, ReturnType::Type(_, ty)) => quote! {
            #[cfg(not(target_arch = "wasm32"))]
            fn main() -> #ty {
                let _ = ::myth::app::__macro_support::env_logger::try_init();
                ::myth::app::__macro_support::pollster::block_on(#user_ident())
            }
        },
    }
}

fn wasm_wrapper(user_ident: &syn::Ident, return_type: &ReturnType, is_async: bool) -> TokenStream {
    let report_result = returns_result(return_type);

    let call = match (is_async, report_result, return_type) {
        (true, true, _) => quote! {
            ::myth::app::__macro_support::wasm_bindgen_futures::spawn_local(async {
                ::myth::app::__macro_support::report_wasm_result(#user_ident().await);
            });
        },
        (true, false, _) => quote! {
            ::myth::app::__macro_support::wasm_bindgen_futures::spawn_local(async {
                let _ = #user_ident().await;
            });
        },
        (false, true, _) => quote! {
            ::myth::app::__macro_support::report_wasm_result(#user_ident());
        },
        (false, false, ReturnType::Default) => quote! {
            #user_ident();
        },
        (false, false, ReturnType::Type(_, _)) => quote! {
            let _ = #user_ident();
        },
    };

    quote! {
        #[cfg(target_arch = "wasm32")]
        #[::myth::app::__macro_support::wasm_bindgen::prelude::wasm_bindgen(start)]
        pub fn wasm_main() {
            ::std::panic::set_hook(Box::new(
                ::myth::app::__macro_support::console_error_panic_hook::hook,
            ));
            let _ = ::myth::app::__macro_support::console_log::init_with_level(
                ::myth::app::__macro_support::log::Level::Info,
            );

            #call
        }
    }
}

fn returns_result(return_type: &ReturnType) -> bool {
    let ReturnType::Type(_, ty) = return_type else {
        return false;
    };

    let Type::Path(path) = ty.as_ref() else {
        return false;
    };

    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Result")
}