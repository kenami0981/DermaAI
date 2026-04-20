using Derma.Infrastructure;
using MediatR;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Derma.Application.SkinAnalysis
{
    public class SkinAnalysisDetails
    {
        public class Query : IRequest<Derma.Domain.SkinAnalysis>
        {
            public Guid Id { get; set; }
        }

        public class Handler : IRequestHandler<Query, Derma.Domain.SkinAnalysis>
        {
            private readonly DataContext _context;
            public Handler(DataContext context)
            {
                _context = context;
            }
            public async Task<Derma.Domain.SkinAnalysis> Handle(Query request, CancellationToken cancellationToken)
            {
                return await _context.SkinAnalysis.FindAsync(request.Id);
            }
        }
    }
}

