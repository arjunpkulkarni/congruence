/**
 * IndexedDB-backed offline buffer for in-flight clinical note edits.
 *
 * Contract:
 *   - Every keystroke the user makes is also mirrored here before the network write.
 *   - If the tab crashes / network drops / device goes offline, edits survive.
 *   - Once the server ack is received, the local buffer entry is cleared.
 *
 * Keyed by session_video_id (one draft buffer per session).
 */

const DB_NAME = "congruence_clinical_notes";
const DB_VERSION = 1;
const STORE_NAME = "pending_note_edits";

export interface PendingNoteEdit {
  sessionVideoId: string;
  contentJson: Record<string, unknown>;
  contentMarkdown: string;
  updatedAt: string;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "sessionVideoId" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

export async function bufferNoteEdit(edit: PendingNoteEdit): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put(edit);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (err) {
    console.warn("clinical-notes-store: buffer failed (non-fatal):", err);
  }
}

export async function clearBufferedNoteEdit(sessionVideoId: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(sessionVideoId);
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    db.close();
  } catch (err) {
    console.warn("clinical-notes-store: clear failed (non-fatal):", err);
  }
}

export async function getBufferedNoteEdit(
  sessionVideoId: string
): Promise<PendingNoteEdit | null> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).get(sessionVideoId);
    const result = await new Promise<PendingNoteEdit | null>((resolve, reject) => {
      request.onsuccess = () => resolve((request.result as PendingNoteEdit) || null);
      request.onerror = () => reject(request.error);
    });
    db.close();
    return result;
  } catch {
    return null;
  }
}

export async function getAllBufferedNoteEdits(): Promise<PendingNoteEdit[]> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readonly");
    const request = tx.objectStore(STORE_NAME).getAll();
    const result = await new Promise<PendingNoteEdit[]>((resolve, reject) => {
      request.onsuccess = () => resolve(request.result as PendingNoteEdit[]);
      request.onerror = () => reject(request.error);
    });
    db.close();
    return result;
  } catch {
    return [];
  }
}
