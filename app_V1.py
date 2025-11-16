import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Affari Tuoi Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Session state & helpers
# -----------------------

def init_session_state():
    """Initialize all session state variables"""
    if 'initialized' not in st.session_state:
        # Create initial prize list with exact values from specification
        prizes = [
            # Blue prizes (low)
            {"label": "0", "value": 0, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "1", "value": 1, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "5", "value": 5, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "10", "value": 10, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "20", "value": 20, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "50", "value": 50, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "75", "value": 75, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "100", "value": 100, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "200", "value": 200, "color": "blue", "opened": False, "is_black_box": False},
            {"label": "500", "value": 500, "color": "blue", "opened": False, "is_black_box": False},
            # Red prizes (high)
            {"label": "Black Box", "value": None, "color": "red", "opened": False, "is_black_box": True},
            {"label": "10.000", "value": 10000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "15.000", "value": 15000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "20.000", "value": 20000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "30.000", "value": 30000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "50.000", "value": 50000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "75.000", "value": 75000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "100.000", "value": 100000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "200.000", "value": 200000, "color": "red", "opened": False, "is_black_box": False},
            {"label": "300.000", "value": 300000, "color": "red", "opened": False, "is_black_box": False},
        ]
        
        st.session_state.remaining_prizes = prizes
        st.session_state.history = []
        st.session_state.current_offer = None
        st.session_state.goal = 0
        st.session_state.step_counter = 0
        st.session_state.box_changed_flag = False
        st.session_state.initialized = True
        
        # Add initial state to history
        add_history_row("initial")

def reset_game():
    """Reset the game to initial state"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def get_remaining_prizes(only_unopened: bool = True) -> List[Dict]:
    """Get list of remaining prizes (optionally only unopened)"""
    if only_unopened:
        return [p for p in st.session_state.remaining_prizes if not p['opened']]
    return st.session_state.remaining_prizes

def get_remaining_known_values() -> List[float]:
    """Get numeric values of remaining prizes (excluding Black Box)"""
    remaining = get_remaining_prizes()
    return [p['value'] for p in remaining if p['value'] is not None]

def get_scenario_values(scenario: str) -> List[float]:
    """Get all remaining prize values for a given scenario (A or B)
    
    Scenario A: Black Box = mean of known prizes
    Scenario B: Black Box = median of known prizes
    """
    remaining = get_remaining_prizes()
    known_values = get_remaining_known_values()
    
    # If no known values remain but Black Box does, treat Black Box as 0.0
    if not known_values:
        black_box_remains = any(p['is_black_box'] for p in remaining)
        if black_box_remains:
            return [0.0]
        return []
    
    # Check if Black Box is still in play
    black_box_remains = any(p['is_black_box'] for p in remaining)
    
    # Start with known values
    values = known_values.copy()
    
    # Add Black Box value if it remains
    if black_box_remains:
        if scenario == 'A':
            black_box_value = np.mean(known_values)
        else:  # scenario == 'B'
            black_box_value = np.median(known_values)
        values.append(black_box_value)
    
    return values

def compute_indicators(scenario: str) -> Dict:
    """Compute all indicators for a given scenario"""
    values = get_scenario_values(scenario)
    remaining = get_remaining_prizes()
    
    if not values:
        return {
            f'EV_{scenario}': 0.0,
            f'Median_{scenario}': 0.0,
            f'Std_{scenario}': 0.0,
            f'P_red_{scenario}': 0.0,
            f'P_le_offer_{scenario}': None,
            f'R_{scenario}': None,
            f'P_ge_goal_{scenario}': None,
        }
    
    # Basic statistics
    indicators = {
        f'EV_{scenario}': float(np.mean(values)),
        f'Median_{scenario}': float(np.median(values)),
        f'Std_{scenario}': float(np.std(values)) if len(values) > 1 else 0.0,
    }
    
    # Probability of red prize:
    # P_red = (# red prizes remaining) / (total prizes remaining)
    red_count = 0
    for p in remaining:
        if p['is_black_box']:
            # Black Box is always considered a red/high prize
            red_count += 1
        elif p['value'] is not None and p['value'] >= 10000:
            red_count += 1
    
    indicators[f'P_red_{scenario}'] = red_count / len(values) if values else 0.0
    
    # Doctor's offer related metrics
    if st.session_state.current_offer is not None and st.session_state.current_offer > 0:
        offer = st.session_state.current_offer
        p_le_offer = sum(1 for v in values if v <= offer) / len(values) if values else 0.0
        indicators[f'P_le_offer_{scenario}'] = p_le_offer
        indicators[f'R_{scenario}'] = offer / indicators[f'EV_{scenario}'] if indicators[f'EV_{scenario}'] > 0 else 0.0
    else:
        indicators[f'P_le_offer_{scenario}'] = None
        indicators[f'R_{scenario}'] = None
    
    # Goal related metrics
    if st.session_state.goal and st.session_state.goal > 0:
        goal = st.session_state.goal
        p_ge_goal = sum(1 for v in values if v >= goal) / len(values) if values else 0.0
        indicators[f'P_ge_goal_{scenario}'] = p_ge_goal
    else:
        indicators[f'P_ge_goal_{scenario}'] = None
    
    return indicators

def count_remaining_by_color() -> Tuple[int, int]:
    """Count remaining blue and red prizes"""
    remaining = get_remaining_prizes()
    blue_count = sum(1 for p in remaining if p['color'] == 'blue')
    red_count = sum(1 for p in remaining if p['color'] == 'red')
    return blue_count, red_count

def add_history_row(action: str, prize_label: Optional[str] = None, prize_value: Optional[float] = None):
    """Add a new row to the history"""
    blue_remaining, red_remaining = count_remaining_by_color()
    
    # Compute indicators for both scenarios
    indicators_A = compute_indicators('A')
    indicators_B = compute_indicators('B')
    
    row = {
        'step': st.session_state.step_counter,
        'action': action,
        'prize_label': prize_label,
        'prize_value': prize_value,
        'num_blue_remaining': blue_remaining,
        'num_red_remaining': red_remaining,
        'EV_A': indicators_A['EV_A'],
        'Median_A': indicators_A['Median_A'],
        'Std_A': indicators_A['Std_A'],
        'EV_B': indicators_B['EV_B'],
        'Median_B': indicators_B['Median_B'],
        'Std_B': indicators_B['Std_B'],
        'offer': st.session_state.current_offer,
        'P_le_offer_A': indicators_A['P_le_offer_A'],
        'P_le_offer_B': indicators_B['P_le_offer_B'],
        'goal': st.session_state.goal if st.session_state.goal > 0 else None,
        'P_ge_goal_A': indicators_A['P_ge_goal_A'],
        'P_ge_goal_B': indicators_B['P_ge_goal_B'],
        'R_A': indicators_A['R_A'],
        'R_B': indicators_B['R_B'],
        'box_changed_flag': st.session_state.box_changed_flag,
        'notes': ''
    }
    
    st.session_state.history.append(row)

def open_prize(prize_idx: int):
    """Open a prize (mark it as opened)"""
    prize = st.session_state.remaining_prizes[prize_idx]
    
    if not prize['opened']:
        prize['opened'] = True
        st.session_state.step_counter += 1
        
        # Determine prize value for history
        if prize['is_black_box']:
            prize_value = "Black Box"
        else:
            prize_value = prize['value']
        
        # Add to history
        add_history_row(
            action="open_prize",
            prize_label=prize['label'],
            prize_value=prize_value
        )

def record_offer():
    """Record the Doctor's offer"""
    st.session_state.step_counter += 1
    add_history_row(action="offer")

def mark_box_change():
    """Mark that the contestant changed box"""
    st.session_state.box_changed_flag = True
    st.session_state.step_counter += 1
    add_history_row(action="box_change")

# -----------------------
# UI components
# -----------------------

def display_blue_prizes(container):
    """Display blue prize buttons in the given container"""
    with container:
        st.subheader("🔵 Blue Prizes (€0 - €500)")
        blue_prizes = [p for p in st.session_state.remaining_prizes if p['color'] == 'blue']
        
        for prize in blue_prizes:
            prize_idx = st.session_state.remaining_prizes.index(prize)
            button_label = f"€{prize['label']}"
            
            # Blue prizes: use secondary style (not red primary)
            button_type = "secondary"
            
            if st.button(
                button_label,
                key=f"prize_blue_{prize_idx}",
                disabled=prize['opened'],
                use_container_width=True,
                type=button_type
            ):
                open_prize(prize_idx)
                st.rerun()

def display_red_prizes(container):
    """Display red prize buttons in the given container"""
    with container:
        st.subheader("🔴 Red Prizes (≥€10.000)")
        red_prizes = [p for p in st.session_state.remaining_prizes if p['color'] == 'red']
        
        for prize in red_prizes:
            prize_idx = st.session_state.remaining_prizes.index(prize)
            
            if prize['is_black_box']:
                button_label = "📦 Black Box"
            else:
                button_label = f"€{prize['label']}"
            
            # Red prizes: use primary style when unopened, secondary when opened
            button_type = "primary" if not prize['opened'] else "secondary"
            
            if st.button(
                button_label,
                key=f"prize_red_{prize_idx}",
                disabled=prize['opened'],
                use_container_width=True,
                type=button_type
            ):
                open_prize(prize_idx)
                st.rerun()

def display_indicators(scenario_display: str):
    """Display current indicators based on scenario selection (center column)"""
    st.subheader("📊 Current Indicators")
    
    blue_remaining, red_remaining = count_remaining_by_color()
    
    # Determine which scenarios to show
    if scenario_display == "Show both scenarios (A and B)":
        scenarios_to_show = ['A', 'B']
        columns = st.columns(2)
    elif scenario_display == "Show only Scenario A":
        scenarios_to_show = ['A']
        columns = [st.container()]
    else:  # Show only Scenario B
        scenarios_to_show = ['B']
        columns = [st.container()]
    
    for i, scenario in enumerate(scenarios_to_show):
        with columns[i]:
            # Title
            if scenario == 'A':
                st.markdown("**📈 Scenario A: Black Box = Mean of known prizes**")
                known_values = get_remaining_known_values()
                if known_values:
                    black_box_value = np.mean(known_values)
                    st.caption(f"Black Box value in this scenario: €{black_box_value:,.0f}")
            else:
                st.markdown("**📊 Scenario B: Black Box = Median of known prizes**")
                known_values = get_remaining_known_values()
                if known_values:
                    black_box_value = np.median(known_values)
                    st.caption(f"Black Box value in this scenario: €{black_box_value:,.0f}")
            
            indicators = compute_indicators(scenario)
            
            # Basic statistics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Expected Value", f"€{indicators[f'EV_{scenario}']:,.0f}")
            with col_b:
                st.metric("Median", f"€{indicators[f'Median_{scenario}']:,.0f}")
            with col_c:
                st.metric("Std Dev", f"€{indicators[f'Std_{scenario}']:,.0f}")
            
            # Remaining prizes count
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                st.metric("Blue Remaining", blue_remaining)
            with col_e:
                st.metric("Red Remaining", red_remaining)
            with col_f:
                st.metric("P(Red Prize)", f"{indicators[f'P_red_{scenario}']:.1%}")
            
            # Offer-related metrics
            if st.session_state.current_offer is not None and st.session_state.current_offer > 0:
                st.markdown("---")
                st.markdown(f"**💰 Doctor's Offer: €{st.session_state.current_offer:,.0f}**")
                
                col_g, col_h = st.columns(2)
                with col_g:
                    p_le = indicators[f'P_le_offer_{scenario}']
                    if p_le is not None:
                        st.metric("P(Prize ≤ Offer)", f"{p_le:.1%}")
                with col_h:
                    ratio = indicators[f'R_{scenario}']
                    if ratio is not None:
                        color = "🟢" if ratio >= 1.0 else "🔴"
                        st.metric("Offer/EV Ratio", f"{color} {ratio:.2f}")
            
            # Goal-related metrics
            if st.session_state.goal and st.session_state.goal > 0:
                st.markdown("---")
                st.markdown(f"**🎯 Your Goal: €{st.session_state.goal:,.0f}**")
                p_ge = indicators[f'P_ge_goal_{scenario}']
                if p_ge is not None:
                    st.metric("P(Prize ≥ Goal)", f"{p_ge:.1%}")

def display_history():
    """Display the history table (center column, bottom)"""
    st.subheader("📜 History")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # Select and reorder columns for better readability
        column_order = [
            'step', 'action', 'prize_label', 'prize_value',
            'num_blue_remaining', 'num_red_remaining',
            'EV_A', 'Median_A', 'Std_A',
            'EV_B', 'Median_B', 'Std_B',
            'offer', 'P_le_offer_A', 'P_le_offer_B',
            'goal', 'P_ge_goal_A', 'P_ge_goal_B',
            'R_A', 'R_B', 'box_changed_flag'
        ]
        
        # Ensure all columns exist
        df = df[[col for col in column_order if col in df.columns]]
        
        # Format display
        df_display = df.copy()
        
        # Format numeric columns
        numeric_cols = ['prize_value', 'EV_A', 'Median_A', 'Std_A', 'EV_B', 'Median_B', 'Std_B', 'offer', 'goal']
        for col in numeric_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f"€{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                )
        
        # Format percentage columns
        pct_cols = ['P_le_offer_A', 'P_le_offer_B', 'P_ge_goal_A', 'P_ge_goal_B']
        for col in pct_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
        
        # Format ratio columns
        ratio_cols = ['R_A', 'R_B']
        for col in ratio_cols:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
        
        # Display the dataframe
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.info("No history yet. Start opening prizes!")

# -----------------------
# Main app
# -----------------------

def main():
    """Main app function"""
    init_session_state()
    
    st.title("🎮 Affari Tuoi Calculator")
    st.markdown("*Statistical analysis tool for the Italian TV game show 'Affari Tuoi'*")
    
    # Sidebar: scenario display, game info, reset
    with st.sidebar:
        st.header("⚙️ Controls")
        
        # Scenario display control
        st.subheader("📊 Display Options")
        scenario_display = st.radio(
            "Scenario Display",
            ["Show both scenarios (A and B)", "Show only Scenario A", "Show only Scenario B"],
            index=0,
            help="Choose which Black Box valuation scenarios to display"
        )
        
        st.divider()
        
        # Game info
        st.subheader("ℹ️ Game Info")
        remaining = get_remaining_prizes()
        st.metric("Prizes Remaining", len(remaining))
        st.metric("Prizes Opened", 20 - len(remaining))
        
        st.divider()
        
        # Reset
        if st.button("🔄 Reset Game", use_container_width=True, type="secondary"):
            reset_game()
            st.rerun()
    
    # Main layout: 3 columns (15%, 70%, 15%)
    left_col, center_col, right_col = st.columns([0.15, 0.7, 0.15])
    
    # Left column: blue prizes
    display_blue_prizes(left_col)
    
    # Center column: offer + goal + indicators + box change + history
    with center_col:
        # Offer and goal at the top
        st.subheader("💰 Doctor's Offer & 🎯 Goal")
        
        col_offer, col_goal = st.columns(2)
        with col_offer:
            offer_input = st.number_input(
                "Doctor's offer (€)",
                min_value=0,
                value=st.session_state.current_offer if st.session_state.current_offer else 0,
                step=1000,
                help="Enter the amount offered by the Doctor"
            )
            if st.button("📝 Record Offer", use_container_width=True, type="primary"):
                if offer_input > 0:
                    st.session_state.current_offer = offer_input
                    record_offer()
                    st.rerun()
            if st.session_state.current_offer:
                st.info(f"Current offer: €{st.session_state.current_offer:,.0f}")
        
        with col_goal:
            goal_input = st.number_input(
                "Goal (minimum acceptable prize, €)",
                min_value=0,
                value=st.session_state.goal,
                step=1000,
                help="Set your personal minimum acceptable amount"
            )
            if goal_input != st.session_state.goal:
                st.session_state.goal = goal_input
                st.rerun()
        
        st.markdown("---")
        
        # Current indicators
        display_indicators(scenario_display)
        
        st.markdown("---")
        
        # Box change below indicators
        st.subheader("📦 Box Change")
        col_bc1, col_bc2 = st.columns([2, 1])
        with col_bc1:
            if st.button("🔄 Mark Box Change", use_container_width=True):
                mark_box_change()
                st.rerun()
        with col_bc2:
            if st.session_state.box_changed_flag:
                st.success("✅ Box changed")
        
        st.markdown("---")
        
        # History at the bottom of the central column
        display_history()
    
    # Right column: red prizes
    display_red_prizes(right_col)

if __name__ == "__main__":
    main()
