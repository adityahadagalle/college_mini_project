import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertOctagon, RotateCcw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('Uncaught error in UI component:', error, errorInfo);
  }

  private handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  public render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="p-8 max-w-lg mx-auto my-12 bg-gradient-to-b from-[#141720]/90 to-[#0D0F14]/95 border border-white/[0.08] shadow-[0_4px_24px_-2px_rgba(0,0,0,0.6),inset_0_1px_0_0_rgba(255,255,255,0.06)] rounded-xl backdrop-blur-md text-center space-y-4">
          <div className="w-10 h-10 mx-auto rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400">
            <AlertOctagon className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-zinc-100">Something went wrong</h3>
            <p className="text-xs text-zinc-400 mt-1">
              An unhandled exception occurred in the component rendering lifecycle.
            </p>
            {this.state.error && (
              <pre className="mt-3 p-3 bg-[#07080C] rounded-lg text-left text-[11px] font-mono text-rose-300 overflow-x-auto border border-rose-900/30">
                {this.state.error.message}
              </pre>
            )}
          </div>
          <button
            onClick={this.handleReset}
            className="inline-flex items-center gap-2 px-3.5 py-2 bg-gradient-to-b from-white via-zinc-100 to-zinc-200 hover:to-white text-zinc-950 rounded-lg text-xs font-semibold shadow-[0_1px_2px_rgba(0,0,0,0.4),0_0_12px_rgba(255,255,255,0.15),inset_0_1px_0_rgba(255,255,255,1)] transition-all active:scale-[0.98]"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reload Component
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
