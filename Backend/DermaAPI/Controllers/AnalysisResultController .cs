using Derma.Application.AnalysisResult;
using Derma.Application.AnalysisResult;
using Derma.Application.SkinAnalysis;
using Derma.Domain;
using Derma.Infrastructure;
using MediatR;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace Derma.API.Controllers
{
    
    public class AnalysisResultController : BaseApiController
    {
        private readonly IMediator _mediator;
        public AnalysisResultController(IMediator mediator)
        {
            _mediator = mediator;
        }
        
        [HttpGet]
        [Authorize(AuthenticationSchemes = "ApiKey")]
        public async Task<ActionResult<List<AnalysisResult>>> GetAnalysisResult() {
            return await _mediator.Send(new AnalysisResultList.Query());
        }
        [HttpGet("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<ActionResult<AnalysisResult>> GetAnalysisResult(Guid id)
        {
            return await Mediator.Send(new AnalysisResultDetails.Query { Id = id });
        }
        [HttpPost]
        [Authorize(AuthenticationSchemes = "ApiKey")]
        public async Task<IActionResult> CreateAnalysisResult(AnalysisResult analysisResult)
        {
            await Mediator.Send(
                new AnalysisResultCreate.Command { AnalysisResult = analysisResult });

            return Ok();
        }

        [HttpDelete("{id}")]
        [Authorize(AuthenticationSchemes = "Bearer,ApiKey")]
        public async Task<IActionResult> DeleteAnalysisResult(Guid id)
        {
            await Mediator.Send(
                new AnalysisResultDelete.Command { Id = id });

            return Ok();
        }
        
    }
}
