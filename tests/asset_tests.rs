//! Asset Storage Tests
//!
//! Tests for:
//! - AssetStorage: add, get, add_with_uuid deduplication
//! - UUID lookup: get_by_uuid, get_handle_by_uuid
//! - Thread safety: concurrent reads via RwLock
//! - AssetServer construction and storage access

use myth::assets::AssetServer;
use myth::assets::storage::AssetStorage;
use myth::resources::Geometry;
use slotmap::new_key_type;
use uuid::Uuid;

new_key_type! { struct TestHandle; }

// ============================================================================
// AssetStorage Basic CRUD
// ============================================================================

#[test]
fn storage_add_and_get() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let handle = storage.add("hello".to_string());
    let value = storage.get(handle).unwrap();
    assert_eq!(&**value, "hello");
}

#[test]
fn storage_get_missing_returns_none() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let handle = storage.add("test".to_string());
    // Remove by dropping and re-creating — we can't remove but can test with a stale handle
    // Instead, just create a new storage and use a handle from the old one
    let storage2 = AssetStorage::<TestHandle, String>::new();
    assert!(storage2.get(handle).is_none());
}

#[test]
fn storage_multiple_assets() {
    let storage = AssetStorage::<TestHandle, i32>::new();
    let h1 = storage.add(10);
    let h2 = storage.add(20);
    let h3 = storage.add(30);

    assert_eq!(*storage.get(h1).unwrap(), 10);
    assert_eq!(*storage.get(h2).unwrap(), 20);
    assert_eq!(*storage.get(h3).unwrap(), 30);
}

// ============================================================================
// UUID-Based Storage
// ============================================================================

#[test]
fn storage_add_with_uuid() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let uuid = Uuid::new_v4();
    let handle = storage.add_with_uuid(uuid, "asset1".to_string());

    let value = storage.get(handle).unwrap();
    assert_eq!(&**value, "asset1");
}

#[test]
fn storage_add_with_uuid_deduplicates() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let uuid = Uuid::new_v4();

    let h1 = storage.add_with_uuid(uuid, "first".to_string());
    let h2 = storage.add_with_uuid(uuid, "second".to_string());

    // Should return same handle
    assert_eq!(h1, h2, "Same UUID should return same handle");

    // Value should be the first one (not overwritten)
    let value = storage.get(h1).unwrap();
    assert_eq!(&**value, "first");
}

#[test]
fn storage_get_by_uuid() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let uuid = Uuid::new_v4();
    storage.add_with_uuid(uuid, "hello".to_string());

    let value = storage.get_by_uuid(&uuid).unwrap();
    assert_eq!(&**value, "hello");
}

#[test]
fn storage_get_by_uuid_missing_returns_none() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let uuid = Uuid::new_v4();
    assert!(storage.get_by_uuid(&uuid).is_none());
}

#[test]
fn storage_get_handle_by_uuid() {
    let storage = AssetStorage::<TestHandle, String>::new();
    let uuid = Uuid::new_v4();
    let handle = storage.add_with_uuid(uuid, "test".to_string());

    let looked_up = storage.get_handle_by_uuid(&uuid).unwrap();
    assert_eq!(handle, looked_up);
}

// ============================================================================
// Thread Safety
// ============================================================================

#[test]
fn storage_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    let storage = Arc::new(AssetStorage::<TestHandle, i32>::new());
    let handle = storage.add(42);

    let mut handles = Vec::new();
    for _ in 0..4 {
        let s = Arc::clone(&storage);
        let h = handle;
        handles.push(thread::spawn(move || {
            let val = s.get(h).unwrap();
            assert_eq!(*val, 42);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

// ============================================================================
// AssetServer Integration
// ============================================================================

#[test]
fn asset_server_stores_geometry() {
    let server = AssetServer::new();
    let geom = Geometry::new_box(1.0, 1.0, 1.0);
    let handle = server.geometries.add(geom);

    let retrieved = server.geometries.get(handle);
    assert!(retrieved.is_some());
}

#[test]
fn asset_server_clone_shares_storage() {
    let server = AssetServer::new();
    let geom = Geometry::new_box(2.0, 2.0, 2.0);
    let handle = server.geometries.add(geom);

    let server2 = server.clone();
    assert!(
        server2.geometries.get(handle).is_some(),
        "Cloned server should share the same storage"
    );
}
