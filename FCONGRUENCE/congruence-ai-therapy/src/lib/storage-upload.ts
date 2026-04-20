import * as tus from "tus-js-client";
import { supabase } from "@/integrations/supabase/client";

const LARGE_FILE_THRESHOLD_BYTES = 64 * 1024 * 1024;
const RESUMABLE_CHUNK_SIZE_BYTES = 6 * 1024 * 1024;
const RESUMABLE_RETRY_DELAYS_MS = [0, 1000, 3000, 5000, 10000];

interface UploadSessionMediaOptions {
  accessToken?: string;
  bucket: string;
  cacheControl?: string;
  file: Blob;
  filePath: string;
  onProgress?: (percent: number, stage: string) => void;
  upsert?: boolean;
}

const shouldUseResumableUpload = (file: Blob) => file.size >= LARGE_FILE_THRESHOLD_BYTES;

const getResumableEndpoint = () => {
  const projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID;
  return projectId
    ? `https://${projectId}.storage.supabase.co/storage/v1/upload/resumable`
    : null;
};

const uploadStandard = async ({
  bucket,
  cacheControl = "3600",
  file,
  filePath,
  upsert = false,
}: UploadSessionMediaOptions) => {
  const { error } = await supabase.storage.from(bucket).upload(filePath, file, {
    cacheControl,
    upsert,
  });

  if (error) {
    throw error;
  }
};

const uploadResumable = async ({
  accessToken,
  bucket,
  cacheControl = "3600",
  file,
  filePath,
  onProgress,
  upsert = false,
}: UploadSessionMediaOptions) => {
  const endpoint = getResumableEndpoint();
  const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

  if (!endpoint || !accessToken || !anonKey) {
    await uploadStandard({ bucket, cacheControl, file, filePath, upsert });
    return;
  }

  onProgress?.(0, "Starting optimized upload...");

  await new Promise<void>((resolve, reject) => {
    const upload = new tus.Upload(file, {
      chunkSize: RESUMABLE_CHUNK_SIZE_BYTES,
      endpoint,
      headers: {
        apikey: anonKey,
        authorization: `Bearer ${accessToken}`,
        "x-upsert": String(upsert),
      },
      metadata: {
        bucketName: bucket,
        objectName: filePath,
        cacheControl,
        contentType: file.type || "application/octet-stream",
      },
      onError: (error) => reject(error),
      onProgress: (bytesUploaded, bytesTotal) => {
        const percent = bytesTotal > 0 ? Math.round((bytesUploaded / bytesTotal) * 100) : 0;
        onProgress?.(percent, "Uploading in chunks...");
      },
      onSuccess: () => {
        onProgress?.(100, "Upload complete");
        resolve();
      },
      removeFingerprintOnSuccess: true,
      retryDelays: RESUMABLE_RETRY_DELAYS_MS,
      uploadDataDuringCreation: true,
    });

    upload
      .findPreviousUploads()
      .then((previousUploads) => {
        if (previousUploads.length > 0) {
          onProgress?.(0, "Resuming upload...");
          upload.resumeFromPreviousUpload(previousUploads[0]);
        }

        upload.start();
      })
      .catch(reject);
  });
};

export const uploadSessionMedia = async (options: UploadSessionMediaOptions) => {
  if (shouldUseResumableUpload(options.file)) {
    await uploadResumable(options);
    return;
  }

  options.onProgress?.(0, "Uploading...");
  await uploadStandard(options);
  options.onProgress?.(100, "Upload complete");
};
