using Derma.Application.CosmeticUsage;
using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Derma.API.Controllers
{
    public class SkinAnalysisController : BaseApiController
    {
        private readonly IMediator _mediator;
        public SkinAnalysisController(IMediator mediator)
        {
            _mediator = mediator;
        }

        [HttpGet]
        public async Task<ActionResult<List<SkinAnalysis>>> GetCosmeticsUsage() {
            return await _mediator.Send(new SkinAnalysisList.Query());
        }
        [HttpGet("{id}")]
        public async Task<ActionResult<SkinAnalysis>> GetSkinAnalysis(Guid id)
        {
            return await Mediator.Send(new SkinAnalysisDetails.Query { Id = id });
        }
    }
}
