def standardize_config(config, field_mappings = {
        'data.normalization': 'data.normalization_range',
        'data.range': 'data.image_range'
        # Add more mappings as needed
    }):
    """
    Standardizes configuration field names using dot notation paths.
    
    Args:
        config: A config object
        
    Returns:
        Config with standardized field names
    """
    
    def get_nested_attr(obj, path):
        """Get nested attribute using dot notation"""
        for attr in path.split('.'):
            obj = getattr(obj, attr)
        return obj
    
    def set_nested_attr(obj, path, value):
        """Set nested attribute using dot notation"""
        attrs = path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        setattr(obj, attrs[-1], value)
    
    def del_nested_attr(obj, path):
        """Delete nested attribute using dot notation"""
        attrs = path.split('.')
        for attr in attrs[:-1]:
            obj = getattr(obj, attr)
        delattr(obj, attrs[-1])

    # Apply all mappings
    for old_path, new_path in field_mappings.items():
        try:
            value = get_nested_attr(config, old_path)
            set_nested_attr(config, new_path, value)
            del_nested_attr(config, old_path)
        except AttributeError:
            # Skip if attribute doesn't exist
            continue
            
    return config