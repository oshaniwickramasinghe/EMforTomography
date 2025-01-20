size=30
    
queue = []
    



# Add elements one by one
for i in range(35):  # Add 35 elements to test the removal behavior

    if len(queue) >= size:
        queue.pop(0)
            
    queue.append(i)
    print(f"Added {i}, Queue state: {queue}")
