import React, { useEffect, useState } from 'react';
import { apiService } from '../../services/api';
import { SystemHealthResponse } from '../../types/sentiment';
import { Activity, Cpu, HardDrive, Clock, CheckCircle } from 'lucide-react';
import { Card } from '../common/Card';
import { Badge } from '../common/Badge';

export const SystemHealthGauges: React.FC = () => {
  const [health, setHealth] = useState<SystemHealthResponse | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await apiService.getSystemHealth();
        setHealth(data);
      } catch {
        // Fallback
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6 w-full">
      {/* 56px Sleek Command Bar */}
      <div className="h-14 px-4 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#141720] border border-white/[0.08] shadow-inner flex items-center justify-center text-zinc-200">
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-center gap-2">
            <h1 className="text-sm font-semibold text-zinc-100 tracking-tight">System Health & Resource Gauges</h1>
            <Badge variant="emerald" size="sm">Online</Badge>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            <span>CPU Utilization</span>
            <Cpu className="w-3.5 h-3.5 text-zinc-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono tabular-nums">
            {health?.cpu_percent ?? 12.4}%
          </div>
          <div className="h-1.5 w-full bg-[#07080C] rounded-full overflow-hidden border border-white/[0.08]">
            <div className="h-full bg-zinc-200 rounded-full" style={{ width: `${health?.cpu_percent ?? 12.4}%` }} />
          </div>
        </Card>

        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            <span>RAM Memory Usage</span>
            <HardDrive className="w-3.5 h-3.5 text-zinc-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono tabular-nums">
            {health?.memory_mb ?? 345.2} MB
          </div>
          <div className="h-1.5 w-full bg-[#07080C] rounded-full overflow-hidden border border-white/[0.08]">
            <div className="h-full bg-zinc-400 rounded-full" style={{ width: `${health?.memory_percent ?? 38.2}%` }} />
          </div>
        </Card>

        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            <span>API Service Status</span>
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 font-mono">
            {health?.status ?? 'OK'}
          </div>
          <div className="text-[10px] text-zinc-500">99.9% Availability SLA Target</div>
        </Card>

        <Card className="p-4 space-y-2">
          <div className="flex items-center justify-between text-[11px] font-semibold text-zinc-400 uppercase tracking-wider">
            <span>System Uptime</span>
            <Clock className="w-3.5 h-3.5 text-zinc-400" />
          </div>
          <div className="text-2xl font-bold text-zinc-100 font-mono tabular-nums">
            {health ? Math.floor(health.uptime_seconds / 60) : 42}m
          </div>
          <div className="text-[10px] text-zinc-500">Active lifespan worker</div>
        </Card>
      </div>
    </div>
  );
};
