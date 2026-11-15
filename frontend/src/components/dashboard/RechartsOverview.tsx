import React from 'react';
import { AnalyticsOverview } from '../../types/sentiment';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Card } from '../common/Card';

interface RechartsOverviewProps {
  data: AnalyticsOverview | null;
}

export const RechartsOverview: React.FC<RechartsOverviewProps> = ({ data }) => {
  const chartData = [
    { name: 'Positive', value: data?.positive_count || 1, color: '#10B981' },
    { name: 'Neutral', value: data?.neutral_count || 1, color: '#F59E0B' },
    { name: 'Negative', value: data?.negative_count || 1, color: '#F43F5E' },
  ];

  return (
    <Card className="p-5 space-y-4">
      <h3 className="text-sm font-semibold text-zinc-200">Sentiment Ratio Spectrum</h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={85}
              paddingAngle={4}
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: '#09090B', borderColor: '#27272A', borderRadius: '8px', color: '#F4F4F5' }}
            />
            <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
};
