#!/usr/bin/env python3
"""
Interactive dashboard example for the Bi-Directional Search Laboratory.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bidirectional_search.visualization.dashboard import InteractiveDashboard


def main():
    """Launch the interactive dashboard."""
    print("Bi-Directional Search Laboratory - Interactive Dashboard")
    print("=" * 60)
    print("Starting dashboard on http://localhost:8050")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Create and run dashboard
        dashboard = InteractiveDashboard(port=8050)
        dashboard.run(debug=False)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Error running dashboard: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
