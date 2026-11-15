import React, { useState, useRef } from 'react';
import { UploadCloud, FileText, Download, CheckCircle, RefreshCw, XCircle, RotateCcw, AlertTriangle } from 'lucide-react';
import { apiService } from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import { BatchJobResponse } from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Badge } from '../common/Badge';

export const FileUploader: React.FC = () => {
  const { showSuccess, showError } = useNotification();

  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<BatchJobResponse | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [cancelled, setCancelled] = useState<boolean>(false);

  const pollRef = useRef<NodeJS.Timeout | null>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const validateAndSetFile = (f: File) => {
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (!['csv', 'txt', 'json'].includes(ext || '')) {
      setError('Invalid file format. Please upload a .csv, .json, or .txt file.');
      return;
    }
    setError(null);
    setFile(f);
    setJob(null);
    setCancelled(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setCancelled(false);

    try {
      const res = await apiService.uploadBatchFile(file);
      setJob(res);
      showSuccess('Batch Job Started', `Job ${res.job_id} launched for asynchronous processing.`);
      pollStatus(res.job_id);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Failed to upload batch file. Please check file format.';
      setError(msg);
      showError('Batch Error', msg);
      setUploading(false);
    }
  };

  const pollStatus = (jobId: string) => {
    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const res = await apiService.getBatchStatus(jobId);
        setJob(res);

        if (res.status === 'COMPLETED') {
          if (pollRef.current) clearInterval(pollRef.current);
          setUploading(false);
          showSuccess('Batch Job Completed', `Successfully processed ${res.processed_rows} records.`);
        } else if (res.status === 'FAILED') {
          if (pollRef.current) clearInterval(pollRef.current);
          setUploading(false);
          showError('Batch Processing Failed', 'The batch job encountered processing errors.');
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
        setUploading(false);
      }
    }, 1500);
  };

  const handleCancel = () => {
    if (pollRef.current) clearInterval(pollRef.current);
    setUploading(false);
    setCancelled(true);
    if (job) setJob({ ...job, status: 'FAILED' });
    showError('Batch Job Cancelled', 'Batch processing job was aborted by user.');
  };

  const progressPercent =
    job && job.total_rows > 0
      ? Math.min(100, Math.round((job.processed_rows / job.total_rows) * 100))
      : job?.status === 'COMPLETED'
      ? 100
      : 0;

  // Calculate ETA remaining seconds
  const remainingRows = job ? Math.max(0, job.total_rows - job.processed_rows) : 0;
  const etaSeconds = Math.ceil(remainingRows / 50);

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <UploadCloud className="w-4 h-4 text-zinc-300" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Asynchronous Batch Ingestion</h1>
            <Badge variant="zinc" size="sm">High-Throughput Streaming</Badge>
          </div>
        </div>
      </div>

      <Card className="p-6 space-y-5">
        {/* Drag & Drop Dropzone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
            isDragOver
              ? 'border-zinc-300 bg-white/[0.05]'
              : file
              ? 'border-emerald-500/60 bg-emerald-950/20'
              : 'border-white/[0.1] hover:border-white/[0.2] bg-[#07080C] shadow-inner'
          }`}
        >
          <input
            type="file"
            accept=".csv,.txt,.json"
            onChange={handleFileSelect}
            id="batch-file-input"
            className="hidden"
          />
          <label htmlFor="batch-file-input" className="cursor-pointer space-y-3 block">
            <div className="w-10 h-10 rounded-xl bg-[#141720] text-zinc-300 flex items-center justify-center mx-auto border border-white/[0.08] shadow-inner">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <p className="text-sm font-semibold text-zinc-200">
                {file ? file.name : 'Drag & Drop CSV, JSON, or TXT file here, or click to browse'}
              </p>
              <p className="text-xs text-zinc-400 mt-1">
                Supports datasets containing text columns (up to 100,000 rows).
              </p>
            </div>
          </label>
        </div>

        {error && (
          <div role="alert" className="p-3 rounded-lg bg-rose-950/30 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        <div className="flex justify-between items-center">
          {file && (
            <span className="text-xs font-mono text-zinc-400 tabular-nums">
              Selected File: <strong className="text-zinc-200">{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
            </span>
          )}

          <div className="flex items-center gap-2 ml-auto">
            {uploading && (
              <Button
                type="button"
                variant="danger"
                size="sm"
                icon={XCircle}
                onClick={handleCancel}
              >
                Cancel Batch
              </Button>
            )}

            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              loading={uploading}
              icon={UploadCloud}
              size="md"
              variant="primary"
            >
              {uploading ? 'Processing Batch...' : 'Start Batch Job'}
            </Button>
          </div>
        </div>

        {/* Progress Bar & Job Status */}
        {job && (
          <div className="pt-4 border-t border-zinc-800 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-zinc-300">
                Status:{' '}
                <span
                  className={`uppercase font-mono font-bold ${
                    job.status === 'COMPLETED'
                      ? 'text-emerald-400'
                      : job.status === 'FAILED'
                      ? 'text-rose-400'
                      : 'text-zinc-200'
                  }`}
                >
                  {cancelled ? 'CANCELLED' : job.status}
                </span>
              </span>
              <span className="text-zinc-400 font-mono tabular-nums">
                {job.processed_rows} / {job.total_rows} rows ({progressPercent}%)
                {uploading && etaSeconds > 0 && ` • ~${etaSeconds}s remaining`}
              </span>
            </div>

            <div className="h-2 w-full bg-[#09090B] rounded-full overflow-hidden p-0.5 border border-zinc-800">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  job.status === 'COMPLETED'
                    ? 'bg-emerald-500'
                    : job.status === 'FAILED'
                    ? 'bg-rose-500'
                    : 'bg-zinc-200'
                }`}
                style={{ width: `${progressPercent}%` }}
              />
            </div>

            {job.status === 'COMPLETED' && (
              <div className="flex justify-between items-center pt-2">
                <span className="text-xs text-emerald-400 flex items-center gap-1.5 font-medium">
                  <CheckCircle className="w-4 h-4" /> Batch processing complete
                </span>
                <a
                  href={apiService.getBatchDownloadUrl(job.job_id)}
                  download
                  className="flex items-center gap-2 bg-zinc-100 hover:bg-white text-zinc-950 text-xs font-semibold px-4 py-2 rounded-lg transition-all shadow-xs"
                >
                  <Download className="w-3.5 h-3.5" /> Download Results CSV
                </a>
              </div>
            )}

            {job.status === 'FAILED' && (
              <div className="flex justify-between items-center pt-2">
                <span className="text-xs text-rose-400 flex items-center gap-1.5 font-medium">
                  <XCircle className="w-4 h-4" /> Processing failed or was cancelled.
                </span>
                <button
                  onClick={handleUpload}
                  className="flex items-center gap-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs font-medium px-3 py-1.5 rounded-lg border border-zinc-800 transition-colors"
                >
                  <RotateCcw className="w-3.5 h-3.5 text-zinc-400" /> Retry Job
                </button>
              </div>
            )}
          </div>
        )}
      </Card>
    </div>
  );
};
