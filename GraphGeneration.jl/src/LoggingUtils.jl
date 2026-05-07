"""
Structured logging utilities for Julia packages.
"""
module LoggingUtils

using Logging

"""
Log an operation with structured context.
"""
function log_operation(level::LogLevel, message::String; 
                     operation::String="", 
                     package::String="", 
                     kwargs...)
    context = Dict(
        "operation" => operation,
        "package" => package,
        kwargs...
    )
    
    @logmsg level message _module=@__MODULE__ _file=@__FILE__ _line=@__LINE__ context...
end

"""
Log info-level message with operation context.
"""
function log_info(message::String; kwargs...)
    log_operation(Logging.Info, message; kwargs...)
end

"""
Log warning-level message with operation context.
"""
function log_warn(message::String; kwargs...)
    log_operation(Logging.Warn, message; kwargs...)
end

"""
Log error-level message with operation context.
"""
function log_error(message::String; kwargs...)
    log_operation(Logging.Error, message; kwargs...)
end

"""
Log debug-level message with operation context.
"""
function log_debug(message::String; kwargs...)
    log_operation(Logging.Debug, message; kwargs...)
end

end # module
