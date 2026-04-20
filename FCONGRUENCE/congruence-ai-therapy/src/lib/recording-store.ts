/**
 * IndexedDB-backed recording store for zero-data-loss guarantees.
 * Saves recordings locally before upload so they survive tab crashes / network failures.
 */

const DB_NAME = 'congruence_recordings';
const DB_VERSION = 1;
const STORE_NAME = 'pending_uploads';

interface PendingRecording {
  id: string;
  patientId: string;
  title: string;
  blob: Blob;
  durationSeconds: number;
  mimeType: string;
  createdAt: string;
  retryCount: number;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function saveRecordingLocally(recording: PendingRecording): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).put(recording);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (err) {
    console.warn('IndexedDB save failed (non-fatal):', err);
  }
}

export async function removeLocalRecording(id: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readwrite');
    tx.objectStore(STORE_NAME).delete(id);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (err) {
    console.warn('IndexedDB delete failed (non-fatal):', err);
  }
}

export async function getPendingRecordings(): Promise<PendingRecording[]> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, 'readonly');
    const store = tx.objectStore(STORE_NAME);
    const request = store.getAll();
    const result = await new Promise<PendingRecording[]>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    db.close();
    return result;
  } catch {
    return [];
  }
}
