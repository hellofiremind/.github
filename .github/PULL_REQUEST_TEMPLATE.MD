https://firemind.atlassian.net/browse/${{ github.head_ref | regex_replace('^.*?([A-Z]+-\d+).*$', '$1') || 'TICKET_NUMBER' }}
