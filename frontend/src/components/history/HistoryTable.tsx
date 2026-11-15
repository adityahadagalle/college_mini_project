import React, { useEffect, useState } from 'react';
import { History, Search, Download, ChevronLeft, ChevronRight, X, Eye, ArrowUpDown } from 'lucide-react';
import { apiService } from '../../services/api';
import { useNotification } from '../../context/NotificationContext';
import { PredictResponse } from '../../types/sentiment';
import { SkeletonLoader } from '../common/SkeletonLoader';
import { EmptyState } from '../common/EmptyState';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';
import { Button } from '../common/Button';

export const HistoryTable: React.FC = () => {
  const { showSuccess, showError } = useNotification();

  const [history, setHistory] = useState<PredictResponse[]>([]);
  const [search, setSearch] = useState<string>('');
  const [sentimentFilter, setSentimentFilter] = useState<string>('ALL');
  const [page, setPage] = useState<number>(1);
  const [pageSize] = useState<number>(10);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedRecord, setSelectedRecord] = useState<PredictResponse | null>(null);
  const [sortField, setSortField] = useState<'created_at' | 'confidence' | 'latency_ms'>('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const offset = (page - 1) * pageSize;
      const data = await apiService.getHistory(search, sentimentFilter, pageSize, offset);
      setHistory(data);
    } catch {
      showError('History Error', 'Failed to retrieve prediction history records.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [sentimentFilter, page]);

  // Esc key listener for modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && selectedRecord) {
        setSelectedRecord(null);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedRecord]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchHistory();
  };

  const handleSort = (field: 'created_at' | 'confidence' | 'latency_ms') => {
    if (sortField === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const sortedHistory = [...history].sort((a, b) => {
    let valA = a[sortField];
    let valB = b[sortField];

    if (sortField === 'created_at') {
      valA = new Date(a.created_at).getTime();
      valB = new Date(b.created_at).getTime();
    }

    if (valA < valB) return sortOrder === 'asc' ? -1 : 1;
    if (valA > valB) return sortOrder === 'asc' ? 1 : -1;
    return 0;
  });

  const handleExportCsv = () => {
    const url = apiService.getHistoryExportUrl();
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentiment_history_export.csv';
    a.click();
    showSuccess('Export Triggered', 'Downloading structured CSV history from backend API.');
  };

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <History className="w-4 h-4 text-zinc-300" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">Prediction Audit Trail</h1>
            <Badge variant="zinc" size="sm">Persistent History</Badge>
          </div>
        </div>

        <Button
          onClick={handleExportCsv}
          variant="secondary"
          size="sm"
          icon={Download}
        >
          Export CSV History
        </Button>
      </div>

      {/* Search & Filter Bar */}
      <Card className="p-5 space-y-4">
        <form onSubmit={handleSearchSubmit} className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-3 top-3" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search prediction review text..."
              className="w-full bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg pl-9 pr-3 py-2 text-xs text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-400 font-sans"
            />
          </div>

          <select
            value={sentimentFilter}
            onChange={(e) => {
              setSentimentFilter(e.target.value);
              setPage(1);
            }}
            className="bg-[#07080C] border border-white/[0.08] shadow-inner rounded-lg px-3 py-2 text-xs text-zinc-300 focus:outline-none focus:border-zinc-400 font-sans cursor-pointer"
          >
            <option value="ALL">All Sentiments</option>
            <option value="POSITIVE">Positive Only</option>
            <option value="NEUTRAL">Neutral Only</option>
            <option value="NEGATIVE">Negative Only</option>
          </select>

          <Button type="submit" variant="primary" size="sm">
            Filter
          </Button>
        </form>

        {/* Table Content */}
        <div className="overflow-x-auto rounded-xl border border-white/[0.08] bg-[#07080C]/40 backdrop-blur-sm shadow-inner">
          <table className="w-full text-left text-xs">
            <thead className="bg-[#090A0F]/90 text-zinc-400 uppercase font-mono text-[10px] border-b border-white/[0.08]">
              <tr>
                <th
                  onClick={() => handleSort('created_at')}
                  className="py-3 px-4 cursor-pointer hover:text-zinc-200 transition-colors"
                >
                  <div className="flex items-center gap-1">
                    Timestamp <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4">Review Content</th>
                <th className="py-3 px-4">Sentiment</th>
                <th
                  onClick={() => handleSort('confidence')}
                  className="py-3 px-4 cursor-pointer hover:text-zinc-200 transition-colors"
                >
                  <div className="flex items-center gap-1">
                    Confidence <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('latency_ms')}
                  className="py-3 px-4 cursor-pointer hover:text-zinc-200 transition-colors"
                >
                  <div className="flex items-center gap-1">
                    Latency <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th className="py-3 px-4 text-right">Inspect</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/[0.04] text-zinc-300">
              {loading ? (
                <tr>
                  <td colSpan={6} className="py-6 px-4">
                    <SkeletonLoader variant="table" count={1} />
                  </td>
                </tr>
              ) : sortedHistory.length === 0 ? (
                <tr>
                  <td colSpan={6} className="py-6">
                    <EmptyState
                      title="No Matching Records"
                      description="No prediction records matched your search query or sentiment filter."
                      actionLabel="Clear Filters"
                      onAction={() => {
                        setSearch('');
                        setSentimentFilter('ALL');
                        setPage(1);
                      }}
                    />
                  </td>
                </tr>
              ) : (
                sortedHistory.map((row) => (
                  <tr
                    key={row.id}
                    onClick={() => setSelectedRecord(row)}
                    className="hover:bg-zinc-800/30 transition-colors cursor-pointer group"
                  >
                    <td className="py-3 px-4 font-mono text-zinc-400 tabular-nums">
                      {new Date(row.created_at).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 max-w-xs truncate font-medium text-zinc-200" title={row.text_content}>
                      {row.text_content}
                    </td>
                    <td className="py-3 px-4">
                      <Badge
                        variant={
                          row.sentiment === 'POSITIVE'
                            ? 'positive'
                            : row.sentiment === 'NEGATIVE'
                            ? 'negative'
                            : 'neutral'
                        }
                        size="sm"
                      >
                        {row.sentiment}
                      </Badge>
                    </td>
                    <td className="py-3 px-4 font-mono text-zinc-200 font-bold tabular-nums">
                      {(row.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-4 font-mono text-zinc-400 tabular-nums">{row.latency_ms} ms</td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedRecord(row);
                        }}
                        aria-label="View record details"
                        className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800 transition-colors"
                      >
                        <Eye className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Bar */}
        <div className="flex justify-between items-center pt-3 border-t border-zinc-800 text-xs text-zinc-400 tabular-nums">
          <span>
            Page <strong className="text-zinc-200">{page}</strong> (Showing {sortedHistory.length} items)
          </span>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 disabled:opacity-30 text-zinc-200 transition-colors"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => setPage((p) => p + 1)}
              disabled={sortedHistory.length < pageSize}
              className="p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 hover:bg-zinc-800 disabled:opacity-30 text-zinc-200 transition-colors"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </Card>

      {/* Record Detail Modal */}
      {selectedRecord && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 bg-[#09090B]/80 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in"
        >
          <Card className="p-6 w-full max-w-lg space-y-4">
            <div className="flex justify-between items-center border-b border-zinc-800 pb-3">
              <h3 className="text-sm font-semibold text-zinc-100">Prediction Record Inspection</h3>
              <button
                onClick={() => setSelectedRecord(null)}
                className="p-1 text-zinc-400 hover:text-white rounded-lg hover:bg-zinc-800"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="space-y-1">
                <span className="text-zinc-400 font-semibold">Review Text Content:</span>
                <p className="p-3 bg-[#09090B] rounded-lg border border-zinc-800 text-zinc-200 leading-relaxed font-sans">
                  "{selectedRecord.text_content}"
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3 pt-2">
                <div className="p-3 rounded-lg bg-[#161920] border border-zinc-800">
                  <span className="text-zinc-400 block mb-1">Sentiment Label</span>
                  <Badge
                    variant={
                      selectedRecord.sentiment === 'POSITIVE'
                        ? 'positive'
                        : selectedRecord.sentiment === 'NEGATIVE'
                        ? 'negative'
                        : 'neutral'
                    }
                    size="sm"
                  >
                    {selectedRecord.sentiment}
                  </Badge>
                </div>
                <div className="p-3 rounded-lg bg-[#161920] border border-zinc-800">
                  <span className="text-zinc-400 block mb-1">Confidence Score</span>
                  <span className="font-mono text-zinc-100 font-bold tabular-nums">
                    {(selectedRecord.confidence * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <div className="pt-2 border-t border-zinc-800 flex justify-between text-zinc-400 font-mono text-[11px] tabular-nums">
                <span>Inference Latency: <strong className="text-zinc-200">{selectedRecord.latency_ms} ms</strong></span>
                <span>ID: {selectedRecord.id?.substring(0, 8)}...</span>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
};
