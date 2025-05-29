import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
# Alright, let's set some cool default settings for our plots so they look nice!
plt.rcParams.update({
    'figure.dpi': 300,  
    'savefig.dpi': 300,
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})


NUM_SESSIONS = 5000
NUM_USERS = 1000
NUM_SHOWS = 500
NUM_GENRES = 4
NUM_DEVICES = 3
DAYS_IN_MONTH = 30

# Genre and device mappings
GENRE_MAP = {
    1: "Drama",
    2: "Comedy",
    3: "Documentary",
    4: "Action"
}

DEVICE_MAP = {
    1: "Mobile",
    2: "TV",
    3: "Laptop"
}

# Add device color map as a global constant at the top of the file after other constants
DEVICE_COLORS = {
    "Mobile": "#ff9999",  # Light red
    "TV": "#66b3ff",      # Light blue
    "Laptop": "#99ff99"   # Light green
}

def validate_dataset(data):
    """
    Validate the shape and content types of the generated dataset.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Check if all required keys exist
        required_keys = ['user_ids', 'show_ids', 'watch_durations', 
                        'genre_ids', 'device_types', 'days', 'is_completed']
        for key in required_keys:
            if key not in data:
                print(f"Error: Missing key '{key}' in dataset")
                return False
        
        # Check shape consistency
        expected_shape = (NUM_SESSIONS,)
        for key, array in data.items():
            if array.shape != expected_shape:
                print(f"Error: Invalid shape for {key}. Expected {expected_shape}, got {array.shape}")
                return False
        
        # Validate data types and ranges
        if not np.all((data['user_ids'] >= 1) & (data['user_ids'] <= NUM_USERS)):
            print("Error: User IDs out of range")
            return False
            
        if not np.all((data['show_ids'] >= 1) & (data['show_ids'] <= NUM_SHOWS)):
            print("Error: Show IDs out of range")
            return False
            
        if not np.all((data['watch_durations'] >= 10) & (data['watch_durations'] <= 180)):
            print("Error: Watch durations out of range")
            return False
            
        if not np.all(np.isin(data['genre_ids'], list(GENRE_MAP.keys()))):
            print("Error: Invalid genre IDs")
            return False
            
        if not np.all(np.isin(data['device_types'], list(DEVICE_MAP.keys()))):
            print("Error: Invalid device types")
            return False
            
        if not np.all((data['days'] >= 1) & (data['days'] <= DAYS_IN_MONTH)):
            print("Error: Days out of range")
            return False
            
        if not np.all(np.isin(data['is_completed'], [0, 1])):
            print("Error: Invalid completion status")
            return False
        
        print("Data validation successful!")
        return True
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        return False

def prepare_dataset(data):
    """
    Prepare the dataset by adding derived features and ensuring data quality.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Updated dataset with additional features
    """
    # Create a copy of the data to avoid modifying the original
    prepared_data = data.copy()
    
    # Add binge-watching feature (1 if duration > 60 minutes, else 0)
    prepared_data['is_binge'] = np.where(data['watch_durations'] > 60, 1, 0)
    
    return prepared_data

def generate_dataset():
    """
    Generate synthetic Netflix viewing data for 5000 sessions.
    
    Returns:
        dict: A dictionary containing numpy arrays for each feature:
            - user_ids: Array of user IDs (1-1000)
            - show_ids: Array of show IDs (1-500)
            - watch_durations: Array of watch durations in minutes
            - genre_ids: Array of genre IDs (1-4)
            - device_types: Array of device types (1-3)
            - days: Array of days (1-30)
            - is_completed: Array of completion status (0 or 1)
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate random user IDs (1-1000)
    user_ids = np.random.randint(1, NUM_USERS + 1, NUM_SESSIONS)
    
    # Generate random show IDs (1-500)
    show_ids = np.random.randint(1, NUM_SHOWS + 1, NUM_SESSIONS)
    
    # Generate watch durations (10-180 minutes)
    # Using normal distribution with mean=60 and std=30, clipped to realistic range
    watch_durations = np.clip(
        np.random.normal(60, 30, NUM_SESSIONS),
        10, 180
    ).astype(int)
    
    # Generate genre IDs (1-4) with some bias towards popular genres
    genre_probs = [0.35, 0.30, 0.15, 0.20]  # Drama, Comedy, Documentary, Action
    genre_ids = np.random.choice(
        list(GENRE_MAP.keys()),
        NUM_SESSIONS,
        p=genre_probs
    )
    
    # Generate device types (1-3) with some bias towards mobile
    device_probs = [0.45, 0.30, 0.25]  # Mobile, TV, Laptop
    device_types = np.random.choice(
        list(DEVICE_MAP.keys()),
        NUM_SESSIONS,
        p=device_probs
    )
    
    # Generate days (1-30)
    days = np.random.randint(1, DAYS_IN_MONTH + 1, NUM_SESSIONS)
    
    # Generate completion status (0 or 1)
    # Probability of completion depends on watch duration
    completion_probs = np.where(watch_durations > 45, 0.7, 0.3)
    is_completed = np.random.binomial(1, completion_probs)
    
    return {
        'user_ids': user_ids,
        'show_ids': show_ids,
        'watch_durations': watch_durations,
        'genre_ids': genre_ids,
        'device_types': device_types,
        'days': days,
        'is_completed': is_completed
    }

def analyze_genre_metrics(data):
    """
    Analyze genre-based metrics including total watch time and average duration.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing genre metrics
    """
    genre_metrics = {}
    
    for genre_id, genre_name in GENRE_MAP.items():
        # Create mask for current genre
        genre_mask = data['genre_ids'] == genre_id
        
        # Calculate metrics
        total_watch_time = np.sum(data['watch_durations'][genre_mask])
        avg_duration = np.mean(data['watch_durations'][genre_mask])
        session_count = np.sum(genre_mask)
        
        genre_metrics[genre_name] = {
            'total_watch_time': total_watch_time,
            'avg_duration': avg_duration,
            'session_count': session_count
        }
    
    return genre_metrics

def plot_genre_distribution(data):
    """
    Create bar charts for genre distribution and metrics.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13))
    fig.suptitle('Netflix Genre Analysis', fontsize=16, y=1.01)
    
    # Get genre metrics
    genre_metrics = analyze_genre_metrics(data)
    
    # Prepare data for plotting
    genres = list(genre_metrics.keys())
    session_counts = [metrics['session_count'] for metrics in genre_metrics.values()]
    avg_durations = [metrics['avg_duration'] for metrics in genre_metrics.values()]
    
    # Plot 1: Session Count by Genre
    bars1 = ax1.bar(genres, session_counts, color='skyblue')
    ax1.set_title('Number of Viewing Sessions by Genre')
    ax1.set_ylabel('Number of Sessions')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # Plot 2: Average Duration by Genre
    bars2 = ax2.bar(genres, avg_durations, color='lightgreen')
    ax2.set_title('Average Watch Duration by Genre')
    ax2.set_ylabel('Average Duration (minutes)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot with consistent settings
    plt.savefig('genre_analysis.png', bbox_inches='tight', dpi=300, pad_inches=0.8) # Increased pad_inches
    plt.close()

def analyze_daily_engagement(data):
    """
    Analyze daily engagement metrics including watch time and completion rates.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing daily metrics
    """
    daily_metrics = {}
    
    # Initialize metrics for each day
    for day in range(1, DAYS_IN_MONTH + 1):
        day_mask = data['days'] == day
        
        # Calculate daily metrics
        total_watch_time = np.sum(data['watch_durations'][day_mask])
        avg_watch_time = np.mean(data['watch_durations'][day_mask])
        total_sessions = np.sum(day_mask)
        completed_sessions = np.sum(data['is_completed'][day_mask])
        partial_sessions = total_sessions - completed_sessions
        
        daily_metrics[day] = {
            'total_watch_time': total_watch_time,
            'avg_watch_time': avg_watch_time,
            'total_sessions': total_sessions,
            'completed_sessions': completed_sessions,
            'partial_sessions': partial_sessions,
            'completion_rate': (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
        }
    
    return daily_metrics

def plot_engagement_trends(data):
    """
    Create line and stacked bar charts for user engagement trends.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    # Get daily metrics
    daily_metrics = analyze_daily_engagement(data)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17, 16))
    fig.suptitle('Netflix User Engagement Trends', fontsize=16, y=0.98)
    
    # Prepare data for plotting
    days = list(daily_metrics.keys())
    avg_watch_times = [metrics['avg_watch_time'] for metrics in daily_metrics.values()]
    completed_sessions = [metrics['completed_sessions'] for metrics in daily_metrics.values()]
    partial_sessions = [metrics['partial_sessions'] for metrics in daily_metrics.values()]
    
    # Find peak viewership day
    peak_day = days[np.argmax(avg_watch_times)]
    peak_watch_time = max(avg_watch_times)
    
    # Plot 1: Average Daily Watch Time
    ax1.plot(days, avg_watch_times, marker='o', linestyle='-', color='royalblue', linewidth=2)
    ax1.set_title('Average Daily Watch Time')
    ax1.set_xlabel('Day of Month')
    ax1.set_ylabel('Average Watch Time (minutes)')
    ax1.grid(True, alpha=0.3)
    
    # Add peak viewership annotation
    ax1.annotate(f'Peak Viewership\nDay {peak_day}\n({peak_watch_time:.1f} min)',
                xy=(peak_day, peak_watch_time),
                xytext=(peak_day + 2, peak_watch_time * 0.7),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                bbox=dict(facecolor='white', edgecolor='red', alpha=0.8),
                fontsize=10)
    
    # Add value labels for every 5th point
    for i in range(0, len(days), 5):
        ax1.text(days[i], avg_watch_times[i], f'{avg_watch_times[i]:.1f}',
                ha='center', va='bottom')
    
    # Plot 2: Completion vs Partial Views (Stacked Bar Chart)
    ax2.bar(days, completed_sessions, label='Completed Views', color='green', alpha=0.7)
    ax2.bar(days, partial_sessions, bottom=completed_sessions, 
            label='Partial Views', color='red', alpha=0.7)
    ax2.set_title('Viewing Completion Trends')
    ax2.set_xlabel('Day of Month')
    ax2.set_ylabel('Number of Sessions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add total session count labels on top of bars
    for i in range(len(days)):
        total = completed_sessions[i] + partial_sessions[i]
        ax2.text(days[i], total, f'{total}',
                ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot with consistent settings
    plt.savefig('engagement_trends.png', bbox_inches='tight', dpi=300, pad_inches=1.0) # Increased pad_inches
    plt.close()

def analyze_device_metrics(data):
    """
    Analyze device usage metrics including session counts and average durations.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing device metrics
    """
    device_metrics = {}
    
    for device_id, device_name in DEVICE_MAP.items():
        # Create mask for current device
        device_mask = data['device_types'] == device_id
        
        # Calculate metrics
        total_sessions = np.sum(device_mask)
        total_watch_time = np.sum(data['watch_durations'][device_mask])
        avg_duration = np.mean(data['watch_durations'][device_mask])
        completion_rate = np.mean(data['is_completed'][device_mask]) * 100
        binge_rate = np.mean(data['is_binge'][device_mask]) * 100
        
        device_metrics[device_name] = {
            'total_sessions': total_sessions,
            'total_watch_time': total_watch_time,
            'avg_duration': avg_duration,
            'completion_rate': completion_rate,
            'binge_rate': binge_rate,
            'percentage': (total_sessions / NUM_SESSIONS) * 100
        }
    
    return device_metrics

def plot_device_usage(data):
    """
    Create pie chart and bar chart for device usage analysis.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    # Get device metrics
    device_metrics = analyze_device_metrics(data)
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9)) # Increased figure size
    fig.suptitle('Netflix Device Usage Analysis', fontsize=16, y=1.05)
    
    # Prepare data for plotting
    devices = list(device_metrics.keys())
    session_counts = [metrics['total_sessions'] for metrics in device_metrics.values()]
    avg_durations = [metrics['avg_duration'] for metrics in device_metrics.values()]
    percentages = [metrics['percentage'] for metrics in device_metrics.values()]
    
    # Use consistent device colors from the color map
    colors = [DEVICE_COLORS[device] for device in devices]
    
    # Plot 1: Pie Chart of Session Distribution
    wedges, texts, autotexts = ax1.pie(
        session_counts,
        labels=devices,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        shadow=True
    )
    ax1.set_title('Distribution of Viewing Sessions by Device')
    
    # Make the percentage labels more readable
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')
    
    # Plot 2: Bar Chart of Average Duration
    bars = ax2.bar(devices, avg_durations, color=colors)
    ax2.set_title('Average Watch Duration by Device')
    ax2.set_ylabel('Average Duration (minutes)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')
    
    # Add text box with statistics within ax1 (pie chart subplot)
    stats_text = "Device Usage Statistics:\n\n"
    for device, metrics in device_metrics.items():
        stats_text += f"{device}:\n"
        stats_text += f"  • Sessions: {metrics['total_sessions']:,}\n"
        stats_text += f"  • Completion Rate: {metrics['completion_rate']:.1f}%\n"
        stats_text += f"  • Binge Rate: {metrics['binge_rate']:.1f}%\n\n"
    
    # Add text box with statistics within ax1, positioned below the pie chart
    ax1.text(0.5, -0.3, stats_text, transform=ax1.transAxes, # Adjusted y position further down
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot with consistent settings
    plt.savefig('device_analysis.png', bbox_inches='tight', dpi=300, pad_inches=1.2) # Increased pad_inches
    plt.close()

def analyze_show_performance(data):
    """
    Analyze show performance metrics including watch time and completion rates.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing show metrics
    """
    show_metrics = {}
    
    # Get unique show IDs
    unique_shows = np.unique(data['show_ids'])
    
    for show_id in unique_shows:
        # Create mask for current show
        show_mask = data['show_ids'] == show_id
        
        # Calculate metrics
        total_sessions = np.sum(show_mask)
        total_watch_time = np.sum(data['watch_durations'][show_mask])
        avg_duration = np.mean(data['watch_durations'][show_mask])
        completion_rate = np.mean(data['is_completed'][show_mask]) * 100
        binge_rate = np.mean(data['is_binge'][show_mask]) * 100
        
        # Only include shows with at least 5 sessions for meaningful statistics
        if total_sessions >= 5:
            show_metrics[show_id] = {
                'total_sessions': total_sessions,
                'total_watch_time': total_watch_time,
                'avg_duration': avg_duration,
                'completion_rate': completion_rate,
                'binge_rate': binge_rate
            }
    
    return show_metrics

def get_top_shows(show_metrics, metric='total_watch_time', n=5):
    """
    Get top N shows based on specified metric.
    
    Args:
        show_metrics (dict): Dictionary containing show metrics
        metric (str): Metric to sort by ('total_watch_time' or 'completion_rate')
        n (int): Number of top shows to return
        
    Returns:
        list: List of tuples (show_id, metric_value) for top N shows
    """
    # Sort shows by the specified metric
    sorted_shows = sorted(
        show_metrics.items(),
        key=lambda x: x[1][metric],
        reverse=True
    )
    
    return sorted_shows[:n]

def plot_top_shows(data):
    """
    Create bar charts for top shows by watch time and completion rate.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    # Get show metrics
    show_metrics = analyze_show_performance(data)
    
    # Get top shows for both metrics
    top_watch_time = get_top_shows(show_metrics, 'total_watch_time')
    top_completion = get_top_shows(show_metrics, 'completion_rate')
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13))
    fig.suptitle('Netflix Top Performing Shows', fontsize=16, y=0.95)
    
    # Prepare data for plotting
    show_ids_watch = [f'Show {show_id}' for show_id, _ in top_watch_time]
    watch_times = [metrics['total_watch_time'] for _, metrics in top_watch_time]
    
    show_ids_completion = [f'Show {show_id}' for show_id, _ in top_completion]
    completion_rates = [metrics['completion_rate'] for _, metrics in top_completion]
    
    # Plot 1: Top Shows by Watch Time
    bars1 = ax1.bar(show_ids_watch, watch_times, color='royalblue')
    ax1.set_title('Top 5 Shows by Total Watch Time')
    ax1.set_ylabel('Total Watch Time (minutes)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Plot 2: Top Shows by Completion Rate
    bars2 = ax2.bar(show_ids_completion, completion_rates, color='forestgreen')
    ax2.set_title('Top 5 Shows by Completion Rate')
    ax2.set_ylabel('Completion Rate (%)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    # Add show statistics as text
    stats_text = "Show Performance Statistics:\n\n"
    
    # Add watch time top shows stats
    stats_text += "Top Shows by Watch Time:\n"
    for show_id, metrics in top_watch_time:
        stats_text += f"Show {show_id}:\n"
        stats_text += f"  • Total Watch Time: {metrics['total_watch_time']:,.0f} minutes\n"
        stats_text += f"  • Sessions: {metrics['total_sessions']:,}\n"
        stats_text += f"  • Completion Rate: {metrics['completion_rate']:.1f}%\n"
        stats_text += f"  • Binge Rate: {metrics['binge_rate']:.1f}%\n\n"
    
    # Add completion rate top shows stats
    stats_text += "Top Shows by Completion Rate:\n"
    for show_id, metrics in top_completion:
        stats_text += f"Show {show_id}:\n"
        stats_text += f"  • Completion Rate: {metrics['completion_rate']:.1f}%\n"
        stats_text += f"  • Total Watch Time: {metrics['total_watch_time']:,.0f} minutes\n"
        stats_text += f"  • Sessions: {metrics['total_sessions']:,}\n"
        stats_text += f"  • Binge Rate: {metrics['binge_rate']:.1f}%\n\n"
    
    # Add text box with statistics within ax2, positioned below the bottom bar chart
    ax2.text(0.02, -0.45, stats_text, transform=ax2.transAxes, # Adjusted y position further down
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=9)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot with consistent settings
    plt.savefig('show_performance.png', bbox_inches='tight', dpi=300, pad_inches=1.2) # Increased pad_inches
    plt.close()

def analyze_user_metrics(data):
    """
    Analyze user-level metrics including watch time and viewing patterns.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing user metrics
    """
    user_metrics = {}
    
    # Get unique user IDs
    unique_users = np.unique(data['user_ids'])
    
    for user_id in unique_users:
        # Create mask for current user
        user_mask = data['user_ids'] == user_id
        
        # Calculate metrics
        total_sessions = np.sum(user_mask)
        total_watch_time = np.sum(data['watch_durations'][user_mask])
        avg_duration = np.mean(data['watch_durations'][user_mask])
        completion_rate = np.mean(data['is_completed'][user_mask]) * 100
        binge_rate = np.mean(data['is_binge'][user_mask]) * 100
        
        # Calculate genre preferences
        genre_counts = np.bincount(data['genre_ids'][user_mask], minlength=len(GENRE_MAP) + 1)[1:]
        favorite_genre_id = np.argmax(genre_counts) + 1
        favorite_genre = GENRE_MAP[favorite_genre_id]
        
        # Calculate device preferences
        device_counts = np.bincount(data['device_types'][user_mask], minlength=len(DEVICE_MAP) + 1)[1:]
        favorite_device_id = np.argmax(device_counts) + 1
        favorite_device = DEVICE_MAP[favorite_device_id]
        
        user_metrics[user_id] = {
            'total_sessions': total_sessions,
            'total_watch_time': total_watch_time,
            'avg_duration': avg_duration,
            'completion_rate': completion_rate,
            'binge_rate': binge_rate,
            'favorite_genre': favorite_genre,
            'favorite_device': favorite_device,
            'genre_distribution': dict(zip(GENRE_MAP.values(), genre_counts)),
            'device_distribution': dict(zip(DEVICE_MAP.values(), device_counts))
        }
    
    return user_metrics

def analyze_genre_binge_metrics(data):
    """
    Analyze genre-level binge-watching metrics.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        dict: Dictionary containing genre binge metrics
    """
    genre_binge_metrics = {}
    
    for genre_id, genre_name in GENRE_MAP.items():
        # Create mask for current genre
        genre_mask = data['genre_ids'] == genre_id
        
        # Calculate binge metrics
        total_sessions = np.sum(genre_mask)
        binge_sessions = np.sum(data['is_binge'][genre_mask])
        binge_rate = (binge_sessions / total_sessions * 100) if total_sessions > 0 else 0
        avg_binge_duration = np.mean(data['watch_durations'][genre_mask & data['is_binge'].astype(bool)])
        
        # Calculate completion metrics for binge sessions
        binge_completion_rate = np.mean(data['is_completed'][genre_mask & data['is_binge'].astype(bool)]) * 100
        
        genre_binge_metrics[genre_name] = {
            'total_sessions': total_sessions,
            'binge_sessions': binge_sessions,
            'binge_rate': binge_rate,
            'avg_binge_duration': avg_binge_duration,
            'binge_completion_rate': binge_completion_rate
        }
    
    return genre_binge_metrics

def plot_advanced_metrics(data):
    """
    Create visualizations for advanced metrics including top user and genre analysis.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    # Get metrics
    user_metrics = analyze_user_metrics(data)
    genre_binge_metrics = analyze_genre_binge_metrics(data)
    
    # Find top user by watch time
    top_user_id = max(user_metrics.items(), key=lambda x: x[1]['total_watch_time'])[0]
    top_user = user_metrics[top_user_id]
    
    # Find top genre by binge rate
    top_genre = max(genre_binge_metrics.items(), key=lambda x: x[1]['binge_rate'])
    
    # Create a figure with overall title
    fig = plt.figure(figsize=(18, 17)) # Increased figure size significantly
    fig.suptitle('Netflix Advanced Metrics Analysis', fontsize=16, y=1.0)

    # Create gridspec for more control over subplot layout
    gs = fig.add_gridspec(3, 2, height_ratios=[1.3, 1.5, 0.6], width_ratios=[1, 1]) # Adjusted height ratio for top subplots

    # Top row: Genre and Device Distribution (split into two subplots)
    ax1 = fig.add_subplot(gs[0, 0]) # Top-left subplot for Genre
    ax1_twin = fig.add_subplot(gs[0, 1]) # Top-right subplot for Device

    # Add a common y-axis label for the top subplots
    fig.text(0.025, 0.66, 'Number of Sessions', va='center', rotation='vertical', fontsize=10) # Adjusted y-coordinate slightly

    # Second row: Genre Binge-Watching Analysis
    ax2 = fig.add_subplot(gs[1, :]) # Second row, spanning both columns
    ax2_twin = ax2.twinx() # Twin axis for binge duration

    # Third row: Space for statistics text (invisible axis)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.axis('off') # Turn off axis for text box

    # Plot 1 (ax1): Top User's Genre Distribution
    genres = list(top_user['genre_distribution'].keys())
    genre_values = list(top_user['genre_distribution'].values())

    ax1.bar(genres, genre_values, color='skyblue')
    ax1.set_title(f'Top User (ID: {top_user_id}) Genre Distribution', fontsize=12) # Adjusted title font size
    ax1.grid(True, alpha=0.3)

    # Adding value labels on top of genre bars.
    for i, v in enumerate(genre_values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=8) # Adjusted vertical offset downwards

    # Plot 1 (ax1_twin): Top User's Device Distribution
    devices = list(top_user['device_distribution'].keys())
    device_values = list(top_user['device_distribution'].values())

    ax1_twin.bar(devices, device_values, color='lightgreen')
    ax1_twin.set_title(f'Top User (ID: {top_user_id}) Device Distribution', fontsize=12) # Separate title, adjusted font size
    ax1_twin.grid(True, alpha=0.3)

    # Adding value labels on top of device bars.
    for i, v in enumerate(device_values):
        ax1_twin.text(i, v + 0.1, str(v), ha='center', va='bottom', fontsize=8) # Adjusted vertical offset downwards

    # Plot 2 (ax2): Genre Binge-Watching Analysis
    genres = list(genre_binge_metrics.keys())
    binge_rates = [metrics['binge_rate'] for metrics in genre_binge_metrics.values()]
    binge_durations = [metrics['avg_binge_duration'] for metrics in genre_binge_metrics.values()]

    bars = ax2.bar(genres, binge_rates, color='coral', alpha=0.7, label='Binge Rate')
    line = ax2_twin.plot(genres, binge_durations, color='purple', marker='o', 
                        label='Avg Binge Duration', linewidth=2)

    ax2.set_title('Genre Binge-Watching Analysis')
    ax2.set_ylabel('Binge Rate (%)')
    ax2_twin.set_ylabel('Average Binge Duration (minutes)')

    # Add value labels for bars, adjusted vertical offset
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.2, # Adjusted vertical offset
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=8)

    # Combine legends for the bottom subplot
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    ax2.grid(True, alpha=0.3)

    # Add statistics text within the dedicated text subplot (ax3)
    stats_text = "Advanced Metrics Statistics:\n\n"

    # Top user statistics
    stats_text += f"Top User (ID: {top_user_id}):\n"
    stats_text += f"  • Total Watch Time: {top_user['total_watch_time']:,.0f} minutes\n"
    stats_text += f"  • Total Sessions: {top_user['total_sessions']:,}\n"
    stats_text += f"  • Average Duration: {top_user['avg_duration']:.1f} minutes\n"
    stats_text += f"  • Completion Rate: {top_user['completion_rate']:.1f}%\n"
    stats_text += f"  • Binge Rate: {top_user['binge_rate']:.1f}%\n"
    stats_text += f"  • Favorite Genre: {top_user['favorite_genre']}\n"
    stats_text += f"  • Favorite Device: {top_user['favorite_device']}\n\n"

    # Top genre statistics
    stats_text += f"Top Genre by Binge Rate ({top_genre[0]}):\n"
    stats_text += f"  • Binge rate: {top_genre[1]['binge_rate']:.1f}%\n"
    stats_text += f"  • Total sessions: {top_genre[1]['total_sessions']:,}\n"
    stats_text += f"  • Binge sessions: {top_genre[1]['binge_sessions']:,}\n"
    stats_text += f"  • Average binge duration: {top_genre[1]['avg_binge_duration']:.1f} minutes\n"
    stats_text += f"  • Binge completion rate: {top_genre[1]['binge_completion_rate']:.1f}%\n"

    ax3.text(0.0, 1.0, stats_text, transform=ax3.transAxes, # Positioned at top-left of ax3
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             fontsize=8) # Reduced font size

    # Adjust layout to prevent label cutoff and improve spacing
    fig.tight_layout(pad=4.5)  # Increased padding further

    # Save the plot with consistent settings
    plt.savefig('advanced_metrics.png', bbox_inches='tight', dpi=300, pad_inches=1.0) # Increased pad_inches further
    plt.close()

def generate_insights_and_recommendations(data):
    """
    Generate insights and recommendations based on the analyzed data.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
        
    Returns:
        tuple: (insights, recommendations) containing lists of insights and recommendations
    """
    # Get all necessary metrics
    user_metrics = analyze_user_metrics(data)
    genre_metrics = analyze_genre_metrics(data)
    genre_binge_metrics = analyze_genre_binge_metrics(data)
    device_metrics = analyze_device_metrics(data)
    show_metrics = analyze_show_performance(data)
    
    insights = []
    recommendations = []
    
    # Insight 1: Analyze binge-watching patterns
    binge_rates_by_device = {
        device: metrics['binge_rate'] 
        for device, metrics in device_metrics.items()
    }
    max_binge_device = max(binge_rates_by_device.items(), key=lambda x: x[1])
    min_binge_device = min(binge_rates_by_device.items(), key=lambda x: x[1])
    
    insights.append(
        f"Key Insight 1 - Device Impact on Binge-Watching: Our data shows that users watching on "
        f"{max_binge_device[0]} exhibit a significantly higher binge rate of {max_binge_device[1]:.1f}%. "
        f"This is in stark contrast to {min_binge_device[0]} users, who have a binge rate of only "
        f"{min_binge_device[1]:.1f}%. This strong correlation indicates that device type is a "
        "major factor influencing binge-watching behavior and overall content engagement."
    )
    
    # Insight 2: Analyze genre preferences and completion rates
    genre_completion_rates = {
        genre: metrics['binge_completion_rate']  # Changed from 'completion_rate' to 'binge_completion_rate'
        for genre, metrics in genre_binge_metrics.items()
    }
    max_completion_genre = max(genre_completion_rates.items(), key=lambda x: x[1])
    min_completion_genre = min(genre_completion_rates.items(), key=lambda x: x[1])
    
    insights.append(
        f"Key Insight 2 - Genre Impact on Completion: We observed that {max_completion_genre[0]} content "
        f"boasts the highest completion rate among binge sessions at {max_completion_genre[1]:.1f}%. "
        f"Conversely, {min_completion_genre[0]} content has the lowest binge session completion "
        f"rate at {min_completion_genre[1]:.1f}%. This highlights that certain genres are much "
        "more effective at retaining viewers' attention until the end of a show or movie."
    )
    
    # Recommendation 1: Content Strategy based on genre performance
    top_genre_by_watch_time = max(
        genre_metrics.items(),
        key=lambda x: x[1]['total_watch_time']
    )
    top_genre_by_binge = max(
        genre_binge_metrics.items(),
        key=lambda x: x[1]['binge_rate']
    )
    
    recommendations.append(
        f"Recommendation 1 - Optimize Content Investment: Given that {top_genre_by_watch_time[0]} "
        f"content drives the highest total watch time ({top_genre_by_watch_time[1]['total_watch_time']:,.0f} minutes), "
        f"and {top_genre_by_binge[0]} content leads in binge-watching rate ({top_genre_by_binge[1]['binge_rate']:.1f}%), "
        "we recommend prioritizing investment in these genres. Developing more content similar "
        "to successful titles in these categories can significantly boost overall engagement and retention."
    )
    
    # Recommendation 2: Device-specific content optimization
    device_avg_durations = {
        device: metrics['avg_duration']
        for device, metrics in device_metrics.items()
    }
    max_duration_device = max(device_avg_durations.items(), key=lambda x: x[1])
    min_duration_device = min(device_avg_durations.items(), key=lambda x: x[1])
    
    recommendations.append(
        f"Recommendation 2 - Tailor Content for Devices: Users on {max_duration_device[0]} engage in "
        f"longer viewing sessions (averaging {max_duration_device[1]:.1f} minutes), while {min_duration_device[0]} "
        f"users have shorter sessions (averaging {min_duration_device[1]:.1f} minutes). We suggest optimizing "
        "content formats and lengths based on device. Consider promoting shorter-form content "
        f"or quick bites for {min_duration_device[0]} users and highlighting longer, more immersive content "
        f"for {max_duration_device[0]} users."
    )
    
    # Recommendation 3: Enhance user engagement features
    top_user = max(user_metrics.items(), key=lambda x: x[1]['total_watch_time'])[1]
    top_shows = get_top_shows(show_metrics, 'completion_rate', n=3)
    
    recommendations.append(
        f"Recommendation 3 - Implement Engagement Boosting Features: Analyzing the behavior of our most "
        f"engaged user (ID: {max(user_metrics.items(), key=lambda x: x[1]['total_watch_time'])[0]}, with "
        f"{top_user['total_watch_time']:,.0f} minutes watched) and the high completion rates of top shows "
        f"({top_shows[0][0]}, {top_shows[1][0]}, {top_shows[2][0]} with rates up to {top_shows[0][1]['completion_rate']:.1f}%), "
        "we recommend implementing features proven to enhance engagement. This includes improving "
        "personalized content recommendations, optimizing the autoplay experience for series, and potentially "
        "introducing interactive elements to increase session duration and completion rates across the platform."
    )
    
    # Adding a new insight based on daily trends (Peak viewership)
    daily_metrics = analyze_daily_engagement(data)
    peak_day_info = max(daily_metrics.items(), key=lambda x: x[1]['avg_watch_time'])
    peak_day = peak_day_info[0]
    peak_watch_time = peak_day_info[1]['avg_watch_time']
    
    insights.append(
        f"Key Insight 3 - Peak Engagement Period: User engagement, specifically average watch time, peaks "
        f"around Day {peak_day} of the month, reaching an average of {peak_watch_time:.1f} minutes. "
        "Understanding these peak periods can inform content release schedules and marketing campaigns "
        "to maximize initial viewership and platform activity."
    )
    
    # Adding a new recommendation based on the lowest completion genre
    min_completion_genre_info = min(genre_binge_metrics.items(), key=lambda x: x[1]['binge_completion_rate'])
    min_completion_genre_name = min_completion_genre_info[0]
    min_completion_rate_value = min_completion_genre_info[1]['binge_completion_rate']
    
    recommendations.append(
        f"Recommendation 4 - Improve Content in Low-Completion Genres: For genres with lower completion "
        f"rates like {min_completion_genre_name} ({min_completion_rate_value:.1f}% binge completion rate), "
        "consider strategies to improve viewer retention. This could involve analyzing user drop-off points, "
        "producing content with stronger narrative hooks, or using targeted notifications to re-engage viewers."
    )
    
    return insights, recommendations

def plot_insights_and_recommendations(data):
    """
    Create a visualization of key insights and recommendations.
    
    Args:
        data (dict): Dictionary containing the dataset arrays
    """
    insights, recommendations = generate_insights_and_recommendations(data)
    
    # Create a figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 13))
    fig.suptitle('Netflix Insights and Recommendations', fontsize=16, y=0.95)
    
    # Plot 1: Key Insights
    ax1.axis('off')
    insight_text = "Key User Behavior Insights:\n\n"
    for i, insight in enumerate(insights, 1):
        insight_text += f"{i}. {insight}\n\n"
    
    ax1.text(0.05, 0.95, insight_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='lightblue', alpha=0.3, edgecolor='blue'),
             fontsize=10)
    
    # Plot 2: Recommendations
    ax2.axis('off')
    recommendation_text = "Strategic Recommendations:\n\n"
    for i, recommendation in enumerate(recommendations, 1):
        recommendation_text += f"{i}. {recommendation}\n\n"
    
    ax2.text(0.05, 0.95, recommendation_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(facecolor='lightgreen', alpha=0.3, edgecolor='green'),
             fontsize=10)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=2.0)
    
    # Save the plot with consistent settings
    plt.savefig('insights_and_recommendations.png', bbox_inches='tight', dpi=300, pad_inches=1.2) # Increased pad_inches
    plt.close()

def main():
    """
    Main function to run the Netflix analysis program.
    """
    print("Netflix User Viewing Analysis")
    print("----------------------------")
    
    # Generate the dataset
    data = generate_dataset()
    
    # Validate the dataset
    if not validate_dataset(data):
        print("Dataset validation failed. Exiting...")
        return
    
    # Prepare the dataset with additional features
    prepared_data = prepare_dataset(data)
    
    # Print some basic statistics about the generated data
    print("\nDataset Statistics:")
    print(f"Total number of sessions: {NUM_SESSIONS}")
    print(f"Unique users: {len(np.unique(prepared_data['user_ids']))}")
    print(f"Unique shows: {len(np.unique(prepared_data['show_ids']))}")
    print(f"Average watch duration: {np.mean(prepared_data['watch_durations']):.2f} minutes")
    print(f"Completion rate: {(np.mean(prepared_data['is_completed']) * 100):.2f}%")
    print(f"Binge-watching rate: {(np.mean(prepared_data['is_binge']) * 100):.2f}%")
    
    # Analyze and print genre metrics
    print("\nGenre Analysis:")
    print("-" * 50)
    genre_metrics = analyze_genre_metrics(prepared_data)
    
    for genre_name, metrics in genre_metrics.items():
        print(f"\n{genre_name}:")
        print(f"  Total watch time: {metrics['total_watch_time']:,.0f} minutes")
        print(f"  Average duration: {metrics['avg_duration']:.1f} minutes")
        print(f"  Number of sessions: {metrics['session_count']:,}")
        print(f"  Percentage of total: {(metrics['session_count'] / NUM_SESSIONS * 100):.1f}%")
    
    # Create and save genre distribution plots
    print("\nGenerating genre analysis plots...")
    plot_genre_distribution(prepared_data)
    print("Plots saved as 'genre_analysis.png'")
    
    # Analyze and print daily engagement metrics
    print("\nDaily Engagement Analysis:")
    print("-" * 50)
    daily_metrics = analyze_daily_engagement(prepared_data)
    
    # Print summary of daily engagement
    total_watch_time = sum(metrics['total_watch_time'] for metrics in daily_metrics.values())
    avg_daily_sessions = np.mean([metrics['total_sessions'] for metrics in daily_metrics.values()])
    avg_completion_rate = np.mean([metrics['completion_rate'] for metrics in daily_metrics.values()])
    
    print(f"\nOverall Daily Engagement Metrics:")
    print(f"  Total watch time across all days: {total_watch_time:,.0f} minutes")
    print(f"  Average daily sessions: {avg_daily_sessions:.1f}")
    print(f"  Average daily completion rate: {avg_completion_rate:.1f}%")
    
    # Create and save engagement trend plots
    print("\nGenerating engagement trend plots...")
    plot_engagement_trends(prepared_data)
    print("Plots saved as 'engagement_trends.png'")
    
    # Analyze and print device usage metrics
    print("\nDevice Usage Analysis:")
    print("-" * 50)
    device_metrics = analyze_device_metrics(prepared_data)
    
    for device_name, metrics in device_metrics.items():
        print(f"\n{device_name}:")
        print(f"  Total sessions: {metrics['total_sessions']:,}")
        print(f"  Percentage of total: {metrics['percentage']:.1f}%")
        print(f"  Average duration: {metrics['avg_duration']:.1f} minutes")
        print(f"  Completion rate: {metrics['completion_rate']:.1f}%")
        print(f"  Binge-watching rate: {metrics['binge_rate']:.1f}%")
    
    # Create and save device usage plots
    print("\nGenerating device usage plots...")
    plot_device_usage(prepared_data)
    print("Plots saved as 'device_analysis.png'")
    
    # Analyze and print show performance metrics
    print("\nShow Performance Analysis:")
    print("-" * 50)
    show_metrics = analyze_show_performance(prepared_data)
    
    # Get and print top shows
    top_watch_time = get_top_shows(show_metrics, 'total_watch_time')
    top_completion = get_top_shows(show_metrics, 'completion_rate')
    
    print("\nTop 5 Shows by Total Watch Time:")
    for show_id, metrics in top_watch_time:
        print(f"\nShow {show_id}:")
        print(f"  Total watch time: {metrics['total_watch_time']:,.0f} minutes")
        print(f"  Number of sessions: {metrics['total_sessions']:,}")
        print(f"  Completion rate: {metrics['completion_rate']:.1f}%")
        print(f"  Binge-watching rate: {metrics['binge_rate']:.1f}%")
    
    print("\nTop 5 Shows by Completion Rate:")
    for show_id, metrics in top_completion:
        print(f"\nShow {show_id}:")
        print(f"  Completion rate: {metrics['completion_rate']:.1f}%")
        print(f"  Total watch time: {metrics['total_watch_time']:,.0f} minutes")
        print(f"  Number of sessions: {metrics['total_sessions']:,}")
        print(f"  Binge-watching rate: {metrics['binge_rate']:.1f}%")
    
    # Create and save show performance plots
    print("\nGenerating show performance plots...")
    plot_top_shows(prepared_data)
    print("Plots saved as 'show_performance.png'")
    
    # Analyze and print advanced metrics
    print("\nAdvanced Metrics Analysis:")
    print("-" * 50)
    user_metrics = analyze_user_metrics(prepared_data)
    genre_binge_metrics = analyze_genre_binge_metrics(prepared_data)
    
    # Find and print top user
    top_user_id = max(user_metrics.items(), key=lambda x: x[1]['total_watch_time'])[0]
    top_user = user_metrics[top_user_id]
    
    print(f"\nTop User (ID: {top_user_id}):")
    print(f"  Total watch time: {top_user['total_watch_time']:,.0f} minutes")
    print(f"  Total sessions: {top_user['total_sessions']:,}\n")
    
    # Find and print top genre by binge rate
    top_genre = max(genre_binge_metrics.items(), key=lambda x: x[1]['binge_rate'])
    
    print(f"\nTop Genre by Binge Rate ({top_genre[0]}):")
    print(f"  Binge rate: {top_genre[1]['binge_rate']:.1f}%")
    print(f"  Total sessions: {top_genre[1]['total_sessions']:,}\n")
    
    # Create and save advanced metrics plots
    print("\nGenerating advanced metrics plots...")
    plot_advanced_metrics(prepared_data)
    print("Plots saved as 'advanced_metrics.png'")

    # Generate and print insights and recommendations
    print("\nInsights and Recommendations:")
    print("-" * 50)
    insights, recommendations = generate_insights_and_recommendations(prepared_data)
    
    print("\nKey User Behavior Insights:")
    for i, insight in enumerate(insights, 1):
        print(f"\n{i}. {insight}")
    
    print("\nStrategic Recommendations:")
    for i, recommendation in enumerate(recommendations, 1):
        print(f"\n{i}. {recommendation}")
    
    # Create and save insights visualization
    print("\nGenerating insights and recommendations visualization...")
    plot_insights_and_recommendations(prepared_data)
    print("Plots saved as 'insights_and_recommendations.png'")

if __name__ == "__main__":
    main()
