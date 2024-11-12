import json
from datetime import datetime
import random
from typing import Dict, List
import pytz

def calculate_response_duration(request_time: str, submission_time: str) -> float:
    """Calculate duration between request and submission in seconds."""
    req_time = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
    sub_time = datetime.fromisoformat(submission_time.replace('Z', '+00:00'))
    return (sub_time - req_time).total_seconds()

def count_valid_responses(user_data: Dict) -> int:
    """Count responses that have both request_time and submission_time."""
    return sum(1 for response in user_data.values() 
              if 'request_time' in response and 'submission_time' in response)

def process_lottery(data: Dict) -> str:
    """Process lottery entries based on response counts."""
    lottery_entries = []
    
    for email, user_data in data.items():
        response_count = count_valid_responses(user_data)
        
        # Add emails with 5 or more responses
        if response_count >= 5:
            lottery_entries.append(email)
            
        # Add emails with 10 or more responses two more times
        if response_count >= 10:
            lottery_entries.extend([email] * 2)
    
    # Randomly select winner if there are entries
    if lottery_entries:
        return random.choice(lottery_entries)
    return None

def process_vouchers(data: Dict) -> Dict[str, int]:
    """Process voucher allocation based on response counts and timing."""
    CUTOFF_DATE = datetime.fromisoformat('2024-10-27T00:00:00+00:00')
    vouchers = {}
    
    for email, user_data in data.items():
        valid_responses_pre_cutoff = 0
        valid_responses_post_cutoff = 0
        
        # Filter and count valid responses
        for response in user_data.values():
            if 'request_time' in response and 'submission_time' in response:
                # Check if response duration is at least 20 seconds
                duration = calculate_response_duration(
                    response['request_time'], 
                    response['submission_time']
                )
                
                # if duration >= 20:
                submission_time = datetime.fromisoformat(
                    response['submission_time'].replace('Z', '+00:00')
                )
                
                if submission_time < CUTOFF_DATE:
                    valid_responses_pre_cutoff += 1
                else:
                    valid_responses_post_cutoff += 1
        
        # Calculate vouchers
        total_vouchers = 0
        
        # Pre-cutoff: 1 voucher per 20 responses
        total_vouchers += (valid_responses_pre_cutoff + valid_responses_post_cutoff) // 20
        
        # Post-cutoff: 1 voucher per 10 responses
        total_vouchers += valid_responses_post_cutoff // 10
        
        # Only include users with 20+ total valid responses
        if (valid_responses_pre_cutoff + valid_responses_post_cutoff) >= 20:
            vouchers[email] = total_vouchers
    
    return vouchers

def process_rewards(data_file: str) -> Dict:
    """Main function to process both lottery and vouchers."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    lottery_winner = process_lottery(data)
    voucher_allocations = process_vouchers(data)
    
    # Calculate voucher statistics
    total_vouchers = sum(voucher_allocations.values())
    num_recipients = len(voucher_allocations)
    avg_vouchers = total_vouchers / num_recipients if num_recipients > 0 else 0
    
    return {
        'lottery_winner': lottery_winner,
        'voucher_allocations': voucher_allocations,
        'voucher_stats': {
            'total_vouchers': total_vouchers,
            'num_recipients': num_recipients,
            'avg_vouchers_per_recipient': round(avg_vouchers, 2)
        }
    }

# Example usage
if __name__ == "__main__":
    results = process_rewards("data/user_timing.json")
    print("\nLottery Winner:", results['lottery_winner'])
    
    print("\nVoucher Statistics:")
    stats = results['voucher_stats']
    print(f"Total Vouchers Allocated: {stats['total_vouchers']}")
    print(f"Number of Recipients: {stats['num_recipients']}")
    print(f"Average Vouchers per Recipient: {stats['avg_vouchers_per_recipient']}")
    
    print("\nVoucher Allocations by Email:")
    for email, vouchers in results['voucher_allocations'].items():
        print(f"{email}: {vouchers} vouchers")